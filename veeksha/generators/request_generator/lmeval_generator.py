from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.client import ClientConfig
from veeksha.config.generators.request_generator.lmeval_generator import (
    LmevalRequestGeneratorConfig,
)
from veeksha.core.request_config import RequestConfig
from veeksha.core.response import Response
from veeksha.core.seeding import SeedManager
from veeksha.generators.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from veeksha.lm_eval.api.instance import Instance
from veeksha.lm_eval.evaluator_utils import (
    TaskOutput,
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
)
from veeksha.lm_eval.tasks import Task, TaskManager, get_task_dict
from veeksha.logger import init_logger
from veeksha.types import LMEvalOutputType

logger = init_logger(__name__)


def detect_task_types(tasks: List[str]) -> bool:
    """Auto-detect if tasks are logit-based by examining their OUTPUT_TYPE.

    Args:
        tasks: List of task names to check

    Returns:
        True if all tasks are logit-based (LOGLIKELIHOOD, LOGLIKELIHOOD_ROLLING, or MULTIPLE_CHOICE)
        False if all tasks are generation-based (GENERATE_UNTIL)

    Raises:
        ValueError: If tasks have mixed types or unknown types
    """
    task_manager = TaskManager()
    task_dict = get_task_dict(tasks, task_manager)  # type: ignore

    if not task_dict:
        raise ValueError("Could not resolve any tasks from provided list.")

    task_types = set()
    for task_name, task_obj in task_dict.items():
        output_type = str(task_obj.OUTPUT_TYPE)
        if output_type in [
            str(LMEvalOutputType.LOGLIKELIHOOD),
            str(LMEvalOutputType.LOGLIKELIHOOD_ROLLING),
            str(LMEvalOutputType.MULTIPLE_CHOICE),
        ]:
            task_types.add("logit")
        elif output_type == str(LMEvalOutputType.GENERATE_UNTIL):
            task_types.add("generation")
        else:
            raise ValueError(
                f"Unknown task output type '{output_type}' for task '{task_name}'"
            )

    if len(task_types) > 1:
        raise ValueError(
            f"Mixed task types not supported. Found both logit-based and generation-based tasks. "
            f"Please separate them into different benchmark runs."
        )

    return "logit" in task_types


class LMEvalRequestGenerator:
    def __init__(
        self,
        config: LmevalRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        client_config: ClientConfig,
        seed_manager: SeedManager,
    ):
        self.config = config
        self.limit = self.config.limit
        self.tokenizer = tokenizer
        self.client_config = client_config

        self.seed_manager = seed_manager
        self.interval_rng_factory = self.seed_manager.numpy_factory("interval")
        self.fewshot_rng = self.seed_manager.random("fewshot")

        self.requests_interval_generator = RequestIntervalGeneratorRegistry.get(
            self.config.interval_generator_config.get_type(),
            self.config.interval_generator_config,
            rng=self.interval_rng_factory(),
        )

        self.task_manager = TaskManager()
        self.task_dict = get_task_dict(self.config.tasks, self.task_manager)  # type: ignore
        if not self.task_dict:
            raise ValueError(
                "LMEvalRequestGenerator could not resolve any tasks from provided."
            )

        # some parameters that can be set later or ignored
        self.gen_kwargs = None
        self.write_out = False
        self.log_samples = False
        self.bootstrap_iters = 100000

        self.task_dict = self._adjust_config(self.task_dict)

        # fewshot_rng already initialized via seed manager

        # now generate requests
        self.requests: Dict[str, List[Instance]] = defaultdict(list)
        self.eval_tasks: List[TaskOutput] = []
        self.cloned_requests: List[Instance] = []
        self.limits: List[Optional[int]] = []
        self.generate_requests()

        self.req_idx = 0

        self.responses = []

    @property
    def num_requests(self):
        return len(self.cloned_requests)

    def _adjust_config(self, task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: self._adjust_config(task_obj)},
                }

            else:
                if task_obj.get_config("output_type") == str(
                    LMEvalOutputType.GENERATE_UNTIL
                ):
                    if self.gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=self.gen_kwargs, update=True
                        )

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if self.config.num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        logger.info(
                            f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                        )
                    else:
                        logger.warning(
                            f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {self.config.num_fewshot}"
                        )
                        task_obj.set_config(
                            key="num_fewshot", value=self.config.num_fewshot
                        )
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (
                        default_num_fewshot := task_obj.get_config("num_fewshot")
                    ) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=self.fewshot_rng.randint(0, 2**32 - 1))

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    def generate_requests(self):
        self.eval_tasks = get_task_list(self.task_dict)

        self.limits = []
        for task_output in self.eval_tasks:
            task: Task = task_output.task  # type: ignore

            # Task type validation is now handled by config.is_logit_based() method

            limit = get_sample_size(task, self.limit)
            self.limits.append(limit)
            task.build_all_requests(limit=limit)

            logger.debug(
                f"Generated {len(task.instances)} requests for {task_output.task_name}"
            )

            for instance in task.instances:
                reqtype = instance.request_type
                self.requests[reqtype].append(instance)

        for reqtype, reqs in self.requests.items():
            for req in reqs:
                self.cloned_requests.extend([req] * req.repeats)  # type: ignore

    def get_request(self) -> RequestConfig:
        if self.req_idx >= len(self.cloned_requests):
            raise StopIteration
        req: Instance = self.cloned_requests[self.req_idx]
        dispatch_delay = self.requests_interval_generator.get_next_inter_request_time()
        self.req_idx += 1

        # just need context to send to the model
        if req.request_type == str(LMEvalOutputType.GENERATE_UNTIL):
            context, all_gen_kwargs = req.args  # type: ignore
            context_length = len(self.tokenizer.encode(context))
            max_gen_toks = all_gen_kwargs.get("max_gen_toks", None)
            if max_gen_toks is not None:
                max_context_length = self.config.max_tokens - max_gen_toks
                if context_length > max_context_length:
                    context = self.tokenizer.decode(
                        self.tokenizer.encode(context)[-max_context_length:]
                    )
                    context_length = len(self.tokenizer.encode(context))
                    logger.warning(
                        f"Context length exceeds max tokens limit. Truncated context to {context_length} tokens."
                    )
            return RequestConfig(
                model=self.client_config.model,
                prompt=(context, context_length),
                dispatch_delay=dispatch_delay,
                sampling_params=all_gen_kwargs,
                llm_api=self.client_config.llm_api,
                address_append_value=self.client_config.address_append_value,
                id=self.req_idx - 1,
            )
        elif req.request_type == str(LMEvalOutputType.LOGLIKELIHOOD):
            context, target = req.args  # type: ignore
            context = context + target
            return RequestConfig(
                model=self.client_config.model,
                prompt=(context, len(self.tokenizer.encode(context))),
                dispatch_delay=dispatch_delay,
                sampling_params={
                    "stream": False,
                    "logprobs": True,
                    "echo": True,
                    "max_tokens": 1,
                    "top_logprobs": 20,
                },
                llm_api=self.client_config.llm_api,
                address_append_value=self.client_config.address_append_value,
                id=self.req_idx - 1,
            )
        else:
            raise NotImplementedError(
                f"Request type {req.request_type} not supported yet."
            )

    def parse_logprobs(self, req: Instance, response: Response) -> Tuple[float, bool]:
        """Parse per-token logprobs for completions responses.

        Supports multiple provider formats:
        1) Non-stream OpenAI-compatible dict with keys: tokens, token_logprobs,
           top_logprobs, text_offset. We sum token_logprobs after the context
           boundary and check greediness against top_logprobs.
        2) Non-stream content list: {"content": [{"token", "logprob",
           "top_logprobs": [{"token", "logprob"}, ...]}, ...]}.
        3) Streaming chunks list: {"chunks": [{"logprob" or "token_logprobs",
           "top_logprobs": [...]}, ...]}.

        If the structure is unrecognized, raises a KeyError.

        Args:
            req: The lm-eval request instance that produced the response.
            response: The model response containing text and provider logprobs.

        Returns:
            Tuple of (sum_logprobs, is_greedy) for the generated segment.
        """
        assert response.logprobs is not None
        lp = response.logprobs
        context, _ = req.args  # type: ignore
        ctxlen = len(self.tokenizer.encode(context))

        # Case 1: tokens/token_logprobs arrays
        if "token_logprobs" in lp and "top_logprobs" in lp:
            tokens_logprobs = lp["token_logprobs"][ctxlen:-1]
            top_logprobs = lp["top_logprobs"][ctxlen:-1]
            logprobs_sum = sum(tokens_logprobs)
            is_greedy = True
            for tok_lp, top in zip(tokens_logprobs, top_logprobs):
                # top may be a dict mapping token->logprob
                if isinstance(top, dict):
                    if not top:
                        is_greedy = False
                        break
                    EPS = 1e-8
                    if tok_lp < (max(top.values()) - EPS):
                        is_greedy = False
                        break
                else:
                    # Unexpected structure; conservatively mark non-greedy
                    is_greedy = False
                    break
            return (logprobs_sum, is_greedy)

        # Case 2: content list with per-token objects (non-stream)
        if isinstance(lp.get("content"), list):
            content = lp["content"]
            # Slice off context tokens using ctxlen as an approximate boundary
            sliced = content[ctxlen:]
            logprobs_list: List[float] = []
            greedies: List[bool] = []
            for entry in sliced:
                tok_lp = entry.get("logprob")
                if tok_lp is None:
                    continue
                logprobs_list.append(tok_lp)
                top = entry.get("top_logprobs") or []
                max_top = None
                if isinstance(top, list) and top:
                    # Entries like {"token": str, "logprob": float}
                    try:
                        max_top = max((t.get("logprob", float("-inf")) for t in top))
                    except Exception:
                        max_top = None
                greedies.append(max_top is not None and tok_lp >= max_top)

            logprobs_sum = sum(logprobs_list) if logprobs_list else 0.0
            is_greedy = all(greedies) if greedies else False
            return (logprobs_sum, is_greedy)

        # Case 3: chunks list (streaming-style)
        if isinstance(lp.get("chunks"), list):
            chunks = lp["chunks"]
            chunks_logprobs_list: List[float] = []
            chunks_greedies: List[bool] = []
            for entry in chunks:
                # Some servers may provide either 'logprob' or 'token_logprobs'
                tok_lp = entry.get("logprob")
                if tok_lp is None and isinstance(entry.get("token_logprobs"), list):
                    # Take the generated token's own logprob if provided as a single-element list
                    try:
                        tok_lp = float(entry["token_logprobs"][0])
                    except Exception:
                        tok_lp = None
                if tok_lp is None:
                    continue
                chunks_logprobs_list.append(tok_lp)
                top = entry.get("top_logprobs") or []
                max_top = None
                if isinstance(top, list) and top:
                    try:
                        max_top = max((t.get("logprob", float("-inf")) for t in top))
                    except Exception:
                        max_top = None
                chunks_greedies.append(max_top is not None and tok_lp >= max_top)

            logprobs_sum = sum(chunks_logprobs_list) if chunks_logprobs_list else 0.0
            is_greedy = all(chunks_greedies) if chunks_greedies else False
            return (logprobs_sum, is_greedy)

        # Unsupported structure
        raise KeyError("Unsupported logprobs structure for completions response")

    def sort_responses(self, responses: List[Response]) -> List[Response]:
        return sorted(responses, key=lambda x: x.id)  # type: ignore

    def get_responses(self, responses: List[Response]) -> None:
        responses = self.sort_responses(responses)
        self.responses = responses

        assert len(self.responses) == len(
            self.cloned_requests
        ), f"Number of responses {len(self.responses)} does not match number of requests {len(self.cloned_requests)}"

        # somehow need to add responses to the task instances (but once that is done, we can evaluate)
        for x, req in zip(self.responses, self.cloned_requests):
            if req.request_type == str(LMEvalOutputType.GENERATE_UNTIL):
                req.resps.append(x.text)
            elif req.request_type == str(LMEvalOutputType.LOGLIKELIHOOD):
                req.resps.append(self.parse_logprobs(req, x))
            else:
                raise NotImplementedError(
                    f"Request type {req.request_type} not supported"
                )

    def evaluate(self) -> Dict[str, Any]:
        # assuming that task instances have been updated with responses in correct way
        for task_output, limit in zip(self.eval_tasks, self.limits):
            task: Task = task_output.task  # type: ignore
            task.apply_filters()

            # Pre-process task.instances to group by doc_id
            instances_by_doc_id = defaultdict(list)
            for instance in task.instances:
                instances_by_doc_id[instance.doc_id].append(instance)
            # Sort instances within each group
            for instances in instances_by_doc_id.values():
                instances.sort(key=lambda x: x.idx)
            # iterate over different filters used
            for filter_key in task.instances[0].filtered_resps.keys():
                doc_iterator = task.doc_iterator(limit=limit)
                for doc_id, doc in doc_iterator:
                    requests = instances_by_doc_id[doc_id]
                    metrics = task.process_results(
                        doc, [req.filtered_resps[filter_key] for req in requests]
                    )
                    for metric, value in metrics.items():  # type: ignore
                        task_output.sample_metrics[(metric, filter_key)].append(value)

        # now calculate aggregate metrics
        for task_output in self.eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=self.bootstrap_iters)
        (
            results,
            samples,
            configs,
            versions,
            num_fewshot,
            higher_is_better,
        ) = consolidate_results(self.eval_tasks)

        # Calculate group metrics
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(
                results, versions, self.task_dict
            )
        results_agg, group_agg = prepare_print_tasks(self.task_dict, results)
        subtask_list = get_subtask_list(self.task_dict)

        # collect all highers_is_better values for metrics in the group's subtasks
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if (
                len(task_list) != 0
            ):  # subtask list will list "task_name": [] for solo tasks
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better.keys():
                            _higher_is_better[m] = h

                        if (
                            m in _higher_is_better
                            and _higher_is_better[m] is not None
                            and _higher_is_better[m] != h
                        ):
                            logger.warning(
                                f"Conflicting higher_is_better values for metric {m} in subtasks of group {group}."
                            )
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **(
                {"groups": dict(group_agg.items())}
                if (bool(group_agg) & show_group_table)  # type: ignore
                else {}
            ),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),  # type: ignore
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),  # type: ignore
                        len(task_output.task.eval_docs),  # type: ignore
                    ),
                }
                for task_output, limit in zip(self.eval_tasks, self.limits)
            },
        }

        return results_dict
