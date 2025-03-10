from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.instance import Instance
from lm_eval.evaluator_utils import (
    TaskOutput,
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
)
from lm_eval.tasks import Task, TaskManager, get_task_dict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.config import ClientConfig, LmevalRequestGeneratorConfig
from veeksha.core.request_config import RequestConfig
from veeksha.core.response import Response
from veeksha.logger import init_logger
from veeksha.types import LMEvalOutputType

logger = init_logger(__name__)


class LMEvalRequestGenerator:
    def __init__(
        self,
        config: LmevalRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        client_config: ClientConfig,
    ):
        self.config = config
        self.limit = self.config.limit
        self.tokenizer = tokenizer
        self.client_config = client_config

        self.task_manager = TaskManager()
        self.task_dict = get_task_dict(self.config.tasks, self.task_manager)  # type: ignore

        # some parameters that can be set later or ignored
        self.gen_kwargs = None
        self.write_out = False
        self.log_samples = False
        self.bootstrap_iters = 100000

        self.task_dict = self._adjust_config(self.task_dict)

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
                task_obj.set_fewshot_seed(seed=self.config.seed)

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    def generate_requests(self):
        self.eval_tasks = get_task_list(self.task_dict)

        self.limits = []
        for task_output in self.eval_tasks:
            task: Task = task_output.task  # type: ignore

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
            # TODO: check if this is the right way to handle this
            return None  # type: ignore
        req: Instance = self.cloned_requests[self.req_idx]
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
                    logger.warning
                    (
                        f"Context length exceeds max tokens limit. Truncated context to {context_length} tokens."
                    )
            return RequestConfig(
                model=self.client_config.model,
                prompt=(context, context_length),
                sampling_params=all_gen_kwargs,
                llm_api=self.client_config.llm_api,
                address_append_value=self.client_config.address_append_value,
                id=self.req_idx - 1,
            )
        elif req.request_type == str(LMEvalOutputType.LOGLIKELIHOOD):
            context, target = req.args  # type: ignore
            # TODO: check how to ensure that model generated only required number of tokens
            # also check if total length is within the limit supported by the model
            return RequestConfig(
                model=self.client_config.model,
                prompt=(context, len(self.tokenizer.encode(context))),
                llm_api=self.client_config.llm_api,
                address_append_value=self.client_config.address_append_value,
                id=self.req_idx - 1,
            )
        else:
            raise NotImplementedError(
                f"Request type {req.request_type} not supported yet."
            )

    def parse_logprobs(self, req: Instance, response: Response) -> Tuple[float, int]:
        assert response.logprobs is not None
        context, target = req.args  # type: ignore
        target_tokens = self.tokenizer.tokenize(target)
        logprobs = float("-inf")
        is_greedy = False
        for i, tok in enumerate(target_tokens):
            if i >= len(response.logprobs):
                # allowing for partial matches?
                break
            j = 0
            # check if the token is in the top logprobs
            while j < len(response.logprobs[i]):
                if response.logprobs[i]["top_logprobs"][j]["token"] == tok:
                    break
                j = j + 1
            # if token is found, add the logprob else break
            if j < len(response.logprobs[i]):
                if logprobs == float("-inf"):
                    logprobs = 0
                    is_greedy = True
                if j > 0:
                    is_greedy = False
                logprobs += response.logprobs[i]["top_logprobs"][j]["logprob"]
            else:
                # allowing for partial matches?
                break
        return logprobs, is_greedy

    def sort_responses(self, responses: List[Response]) -> List[Response]:
        return sorted(responses, key=lambda x: x.id)  # type: ignore

    def get_responses(self, responses: List[Response]) -> None:
        responses = self.sort_responses(responses)
        self.responses = responses

        assert len(self.responses) == len(
            self.cloned_requests
        ), "Number of responses does not match number of requests"

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
