# src/instrumented_molmoact.py

import io
import time
from contextlib import redirect_stdout

import numpy as np

from simpler_env.policies.molmoact.molmoact_model import MolmoActInference
from simpler_env.policies.molmoact.molmoact_model_vllm import MolmoActInferenceVLLM


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    return x


class InstrumentedMolmoAct:
    """
    Wrapper around official SimplerEnv MolmoAct policy.

    Keeps the official action-processing logic unchanged while
    capturing generated_text / depth / trace / parsed_action.
    """

    def __init__(
        self,
        saved_model_path,
        policy_setup="google_robot",
        backend="hf",
        quiet=True,
    ):
        self.quiet = quiet
        self.last_debug = {}

        if backend == "hf":
            self.policy = MolmoActInference(
                saved_model_path=saved_model_path,
                policy_setup=policy_setup,
            )

            # HF version parses with self.model.parse_*
            self.parser_target = self.policy.model

        elif backend == "vllm":
            self.policy = MolmoActInferenceVLLM(
                saved_model_path=saved_model_path,
                policy_setup=policy_setup,
            )

            # vLLM version parses with self.parser.parse_*
            self.parser_target = self.policy.parser

        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._install_parse_hooks()

    def _install_parse_hooks(self):
        self._orig_parse_depth = self.parser_target.parse_depth
        self._orig_parse_trace = self.parser_target.parse_trace
        self._orig_parse_action = self.parser_target.parse_action

        def parse_depth_hook(text, *args, **kwargs):
            result = self._orig_parse_depth(text, *args, **kwargs)

            self.last_debug["generated_text"] = text
            self.last_debug["depth"] = to_jsonable(result)

            return result

        def parse_trace_hook(text, *args, **kwargs):
            result = self._orig_parse_trace(text, *args, **kwargs)

            self.last_debug["generated_text"] = text
            self.last_debug["trace"] = to_jsonable(result)

            return result

        def parse_action_hook(text, *args, **kwargs):
            result = self._orig_parse_action(text, *args, **kwargs)

            self.last_debug["generated_text"] = text
            self.last_debug["parsed_action"] = to_jsonable(result)

            return result

        self.parser_target.parse_depth = parse_depth_hook
        self.parser_target.parse_trace = parse_trace_hook
        self.parser_target.parse_action = parse_action_hook

    def reset(self, task_description):
        self.policy.reset(task_description)

    def step(self, image, task_description):
        self.last_debug = {}

        t0 = time.perf_counter()

        # 官方 wrapper 每一步会 print Depth / Trace / Action。
        # 批量实验时关闭这些输出，否则 350 episodes 会刷爆日志。
        if self.quiet:
            with redirect_stdout(io.StringIO()):
                raw_action, action, annotated_image = self.policy.step(
                    image,
                    task_description,
                )
        else:
            raw_action, action, annotated_image = self.policy.step(
                image,
                task_description,
            )

        inference_time = time.perf_counter() - t0

        self.last_debug["raw_action"] = to_jsonable(raw_action)
        self.last_debug["processed_action"] = to_jsonable(action)
        self.last_debug["inference_time_s"] = inference_time

        return raw_action, action, annotated_image

    def visualize_epoch(self, *args, **kwargs):
        return self.policy.visualize_epoch(*args, **kwargs)