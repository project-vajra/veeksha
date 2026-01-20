from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from lm_eval.api.instance import (
    Instance,
)
from lm_eval.api.task import Task
from lm_eval.evaluator_utils import (
    get_task_list,
)
from lm_eval.tasks import (
    TaskManager,
    get_task_dict,
)

from veeksha.config.generator.session import LmevalSessionGeneratorConfig
from veeksha.core.request import Request
from veeksha.core.request_content import (
    TextChannelRequestContent,
)
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import SessionGraph, SessionNode, add_node
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.session.base import BaseSessionGenerator
from veeksha.logger import init_logger
from veeksha.types import ChannelModality, LMEvalOutputType

logger = init_logger(__name__)


class LMEvalSessionGenerator(BaseSessionGenerator):
    """Session generator that emits one session per lm-eval document.

    The generator owns lm-eval Task/Instance objects. The accuracy evaluator binds
    responses back to these instances using request_id.

    Note:
        A single lm-eval document can require multiple model calls (e.g., multiple
        loglikelihood instances for multiple-choice). In those cases, the generator
        emits a single Session containing multiple Requests.
    """

    def __init__(
        self,
        config: LmevalSessionGeneratorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
        max_sessions: int = -1,
    ):
        self.config = config
        self.seed_manager = seed_manager
        self.tokenizer_provider = tokenizer_provider
        self.text_tokenizer = self.tokenizer_provider.for_modality(ChannelModality.TEXT)
        self.max_sessions = max_sessions

        self._fewshot_rng = self.seed_manager.random("fewshot")

        self.task_manager = TaskManager()
        self.task_dict = get_task_dict(self.config.tasks, self.task_manager)  # type: ignore
        if not self.task_dict:
            raise ValueError("LMEvalSessionGenerator could not resolve any tasks.")

        self.task_dict = self._adjust_config(self.task_dict)

        self.eval_tasks = get_task_list(self.task_dict)
        num_tasks = max(1, len(self.eval_tasks))
        per_task_doc_limit: Optional[int]
        if self.max_sessions < 0:
            per_task_doc_limit = None
        else:
            per_task_doc_limit = max(0, int(math.ceil(self.max_sessions / num_tasks)))

        # Build and collect per-task doc ordering.
        task_to_doc_ids: Dict[str, List[int]] = {}
        task_to_instances_by_doc: Dict[str, Dict[int, List[Instance]]] = {}

        for task_output in self.eval_tasks:
            task: Task = task_output.task  # type: ignore
            task.build_all_requests(limit=per_task_doc_limit)

            instances_by_doc_id: Dict[int, List[Instance]] = defaultdict(list)
            for instance in task.instances:
                doc_id = instance.doc_id
                if doc_id is None:
                    continue
                instances_by_doc_id[doc_id].append(instance)
            for instances in instances_by_doc_id.values():
                instances.sort(key=lambda x: x.idx)

            tname = str(task_output.task_name)
            doc_ids: List[int] = [
                doc_id
                for doc_id, _ in task.doc_iterator(limit=per_task_doc_limit)
                if doc_id is not None
            ]
            task_to_doc_ids[tname] = doc_ids
            task_to_instances_by_doc[tname] = instances_by_doc_id

        # Interleave tasks' doc streams (round-robin) and then truncate to max_sessions.
        all_doc_keys: List[Tuple[str, int]] = []
        max_len = max((len(v) for v in task_to_doc_ids.values()), default=0)
        for i in range(max_len):
            for task_output in self.eval_tasks:
                tname = str(task_output.task_name)
                doc_ids = task_to_doc_ids.get(tname, [])
                if i < len(doc_ids):
                    all_doc_keys.append((tname, doc_ids[i]))

        if self.max_sessions >= 0:
            all_doc_keys = all_doc_keys[: self.max_sessions]

        self._doc_keys: List[Tuple[str, int]] = all_doc_keys

        # Effective per-task limits based on selected doc_keys (used by evaluator).
        included_doc_ids_by_task: Dict[str, set[int]] = defaultdict(set)
        for tname, doc_id in self._doc_keys:
            included_doc_ids_by_task[tname].add(doc_id)

        self.limits: List[Optional[int]] = []
        for task_output in self.eval_tasks:
            tname = str(task_output.task_name)
            # Because we select a prefix via round-robin over per-task prefixes,
            # the number of docs included for a task is a valid `doc_iterator(limit=...)`.
            self.limits.append(len(included_doc_ids_by_task.get(tname, set())))

        # request_id -> instance/context length mapping for evaluator
        self.instance_by_request_id: Dict[int, Instance] = {}
        self.ctxlen_tokens_by_request_id: Dict[int, int] = {}

        self._cursor = 0
        self._next_request_id = 0
        self._instances_by_task_doc = task_to_instances_by_doc

    def _adjust_config(self, task_dict: dict) -> dict:
        adjusted_task_dict: dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: self._adjust_config(task_obj)},
                }
                continue

            # override tasks' fewshot values unless task config forces 0
            default_num_fewshot = task_obj.get_config("num_fewshot")
            if default_num_fewshot == 0:
                logger.info(
                    "num_fewshot is 0 for %s in task config; leaving unchanged.",
                    task_name,
                )
            else:
                task_obj.set_config(key="num_fewshot", value=self.config.num_fewshot)

            task_obj.set_fewshot_seed(seed=self._fewshot_rng.randint(0, 2**32 - 1))
            adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    def capacity(self) -> int:
        return len(self._doc_keys)

    def generate_session(self) -> Session:
        if self._cursor >= len(self._doc_keys):
            raise StopIteration

        session_id = self._cursor
        task_name, doc_id = self._doc_keys[self._cursor]
        instances = self._instances_by_task_doc.get(task_name, {}).get(doc_id, [])

        session_graph = SessionGraph()
        requests: Dict[int, Request] = {}

        node_id = 0
        for instance in instances:
            repeats = int(getattr(instance, "repeats", 1) or 1)
            for _ in range(repeats):
                request_id = self._next_request_id
                self._next_request_id += 1

                add_node(session_graph, SessionNode(id=node_id, wait_after_ready=0.0))
                request = self._build_request(
                    request_id=request_id, instance=instance, node_id=node_id
                )
                requests[node_id] = request
                self.instance_by_request_id[request_id] = instance
                node_id += 1

        self._cursor += 1
        return Session(id=session_id, session_graph=session_graph, requests=requests)

    def _build_request(
        self, request_id: int, instance: Instance, node_id: int = 0
    ) -> Request:
        session_context = {
            "node_id": node_id,
            "wait_after_ready": 0.0,
            "parent_nodes": [],
            "history_parent": None,
        }

        channels: Dict[ChannelModality, Any] = {}
        metadata: Dict[str, Any] = {
            "lmeval_request_type": instance.request_type,
        }
        requested_output: Optional[RequestedOutputSpec] = None

        if instance.request_type == str(LMEvalOutputType.GENERATE_UNTIL):
            context, gen_kwargs, *rest = instance.args  # type: ignore
            multimodal_arg = rest[0] if rest and isinstance(rest[0], dict) else {}

            ctxlen = len(self.text_tokenizer.encode(context))
            max_gen_toks = gen_kwargs.get("max_gen_toks")
            target_output_tokens = (
                int(max_gen_toks) if isinstance(max_gen_toks, int) else 0
            )

            channels[ChannelModality.TEXT] = TextChannelRequestContent(
                input_text=context,
                target_prompt_tokens=ctxlen,
            )
            requested_output = RequestedOutputSpec(
                text=TextOutputSpec(target_tokens=target_output_tokens)
            )

            metadata["api_mode"] = "chat"
            metadata["sampling_params"] = dict(gen_kwargs)

        elif instance.request_type == str(LMEvalOutputType.LOGLIKELIHOOD):
            context, target, *rest = instance.args  # type: ignore
            multimodal_arg = rest[0] if rest and isinstance(rest[0], dict) else {}

            # Match lm-eval's token-boundary handling:
            # if `context` ends with spaces, move those spaces to the beginning of
            # `target` so the context/continuation boundary tokenizes correctly
            n_spaces = len(context) - len(context.rstrip())
            if n_spaces > 0:
                target = context[-n_spaces:] + target
                context = context[:-n_spaces]

            ctxlen = len(self.text_tokenizer.encode(context))
            prompt = context + target
            prompt_len = len(self.text_tokenizer.encode(prompt))

            channels[ChannelModality.TEXT] = TextChannelRequestContent(
                input_text=prompt,
                target_prompt_tokens=prompt_len,
            )
            requested_output = RequestedOutputSpec(text=TextOutputSpec(target_tokens=1))
            if multimodal_arg:
                logger.warning(
                    "Ignoring multimodal inputs for loglikelihood request_id=%d",
                    request_id,
                )

            self.ctxlen_tokens_by_request_id[request_id] = ctxlen

            metadata["api_mode"] = "completions"
            metadata["sampling_params"] = {
                "stream": False,
                # Required for loglikelihood scoring: return logprobs for prompt tokens.
                # This matches upstream lm-eval's OpenAI completions behavior.
                "echo": True,
                # OpenAI-compatible completions API expects an integer `logprobs`
                # (how many top tokens to return). `1` is sufficient for
                # loglikelihood scoring and matches upstream lm-eval behavior.
                "logprobs": 1,
                "temperature": 0,
                # `/completions` uses `max_tokens` (not chat's
                # `max_completion_tokens`)
                "max_tokens": 1,
            }

        else:
            raise NotImplementedError(
                f"lm-eval request_type '{instance.request_type}' not supported yet."
            )

        return Request(
            id=request_id,
            channels=channels,
            session_context=session_context,
            metadata=metadata,
            requested_output=requested_output,
        )
