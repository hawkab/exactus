"""Serialize exact-generation results and verification evidence."""

__author__ = "Olshansky Grigory"

from ..domain.finalization import finalize
from .endpoint_http import encode


def _serialize_encoded(result, mode, model):
    """Encode one model-tagged completion and its verification evidence."""
    return encode({"model": model, "exact_compute": True, **_serialize(result, mode)})


def _serialize(result, mode):
    """Render generation metadata, finalization evidence, and token accounting."""
    proof = result.proof
    if proof is None:
        text, proof, exact = finalize(result.transcript, result.operations, mode)
        emitted = any(s.final and s.origin == "model_marker" for s in proof.spans)
    else:
        text, exact = result.text, result.exact_final
        emitted = result.model_final_emitted
    return {
        "text": text,
        "operations": [
            dict(expression=o.expression, value=o.value, final=o.final, format=o.format)
            for o in result.operations
        ],
        "engine_segments": result.engine_segments,
        "sampled_tokens": result.sampled_tokens,
        "known_tokens": result.known_tokens,
        "known_tokens_scope": "standalone_values_tokenizer_estimate_not_rendered_text_length",
        "prefill_tokens": result.prefill_tokens,
        "exact_final": exact,
        "elapsed_s": result.elapsed_s,
        "request_id": result.request_id,
        "parent_id": result.parent_id,
        "finish_reason": result.finish_reason,
        "audit": proof.as_dict(detail=mode == "audit"),
        **proof.verification(),
        "model_final_emitted": emitted,
        "protocol_complete": True,
        "generation_complete": result.finish_reason == "stop",
        "engine_wait_s": result.engine_wait_s,
        "cpu_stage_s": result.cpu_stage_s,
        "prompt_tokens": result.prompt_tokens,
        "sampled_tokens_exact": result.sampled_tokens_exact,
        "mode": mode,
    }
