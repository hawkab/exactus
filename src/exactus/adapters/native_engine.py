"""AsyncLLM Engine port using token IDs, normal prefill and engine-owned MTP."""

import inspect

from ..domain.ports import Segment
from ..prompts import MARKER_HINT
from .engine_client import EngineError


class NativeTokenizer:
    def __init__(self, tokenizer, *, chat_template=None, template_kwargs=None):
        self.inner = tokenizer
        self.chat_template, self.template_kwargs = (
            chat_template,
            dict(template_kwargs or {}),
        )

    def encode(self, text):
        ids = self.inner.encode(text, add_special_tokens=False)
        return ids if isinstance(ids, list) else list(ids)

    def decode(self, ids):
        return self.inner.decode(ids, skip_special_tokens=False)

    def encode_prompt(self, text):
        return self.encode_messages(
            [
                {"role": "system", "content": MARKER_HINT},
                {"role": "user", "content": text},
            ]
        )

    def encode_messages(self, messages):
        kwargs = {
            **self.template_kwargs,
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": False,
            "return_tensors": None,
        }
        if self.chat_template is not None:
            kwargs["chat_template"] = self.chat_template
        return self.inner.apply_chat_template(messages, **kwargs)


class NativeEngine:
    """Known finals skip decoding; intermediates advance state through prefill.

    No patched KV/recurrent buffers and no speculative safety-gate bypass.
    Real GPU qualification remains required.
    """

    def __init__(self, client, *, context_limit=None, sampling_factory=None):
        self.client, self.context_limit = client, context_limit
        self._sampling_factory = sampling_factory

    async def generate(self, prompt_ids, max_tokens, request_id):
        factory = self._sampling_factory
        if factory is None:
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind

            factory = SamplingParams
            output_kind = RequestOutputKind.FINAL_ONLY
        else:
            output_kind = 2  # contract doubles; real vLLM factory above uses its enum
        params = factory(
            max_tokens=max_tokens,
            temperature=0,
            stop=["</calc>", "</calc-final>"],
            include_stop_str_in_output=True,
            output_kind=output_kind,
        )
        complete = False
        if self.client is None:
            raise EngineError("native engine is unavailable")
        stream = self.client.generate(
            {"prompt_token_ids": prompt_ids}, params, request_id
        )
        try:
            latest = None
            async for output in stream:
                latest = output
            if latest is None or not latest.outputs:
                raise EngineError("native engine returned no output")
            choice = latest.outputs[0]
            complete = bool(getattr(latest, "finished", False))
            if not complete:
                raise EngineError("native engine stream ended before completion")
            return Segment(
                choice.text,
                tuple(choice.token_ids),
                len(choice.token_ids),
                choice.finish_reason or "unknown",
            )
        finally:
            try:
                if not complete:
                    aborted = self.client.abort(request_id)
                    if inspect.isawaitable(aborted):
                        await aborted
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
