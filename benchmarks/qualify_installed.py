"""Read candidate sources from an in-memory tar; test installed vLLM on CPU only.

Run with --source-tar SOURCE.tar.gz inside the existing vLLM environment.
Does not install a wheel, write
candidate files, instantiate an LLM, create CUDA tensors or change service state.
The bootstrap provides ``archive`` (tarfile.TarFile backed by BytesIO).
"""

import asyncio
import importlib.abc
import importlib.metadata
import importlib.util
import json
import inspect
import os
import sys
from types import SimpleNamespace as NS

archive = globals().get("archive")
if archive is None:
    import argparse
    import io
    from pathlib import Path
    import tarfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tar", type=Path, required=True)
    args = parser.parse_args()
    if args.source_tar.stat().st_size > 8 * 1024 * 1024:
        parser.error("source archive exceeds 8 MiB")
    archive = tarfile.open(fileobj=io.BytesIO(args.source_tar.read_bytes()))


class Candidate(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name != "exactus" and not name.startswith("exactus."):
            return None
        prefix = "src/" + name.replace(".", "/")
        for suffix, package in (("/__init__.py", True), (".py", False)):
            try:
                archive.getmember(prefix + suffix)
            except KeyError:
                continue
            spec = importlib.util.spec_from_loader(name, self, is_package=package)
            spec.loader_state = prefix + suffix
            return spec
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = module.__spec__.loader_state
        module.__file__ = "<candidate>/" + path
        exec(
            compile(archive.extractfile(path).read(), module.__file__, "exec"),
            module.__dict__,
        )


# Only this short-lived qualification process is changed, never the live server.
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
sys.meta_path.insert(0, Candidate())

import torch
from vllm import SamplingParams
from vllm.plugins.endpoint_plugins.interface import EndpointPlugin
from vllm.sampling_params import RequestOutputKind
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from exactus.adapters.default_plugin import ExactusEndpointPlugin
from exactus import __version__
from exactus.adapters.native_engine import NativeEngine
from exactus.adapters.vllm_logits import LegacyExactusLogitsProcessor

assert isinstance(ExactusEndpointPlugin(), EndpointPlugin)
proc = LegacyExactusLogitsProcessor(
    NS(speculative_config=None), torch.device("cpu"), False
)
history = []
batch_kwargs = {
    "removed": [],
    "added": [
        (0, SamplingParams(extra_args={"exactus_token_ids": [1, 2]}), [], history)
    ],
    "moved": [],
}
if "batch_size" in inspect.signature(BatchUpdate).parameters:
    batch_kwargs["batch_size"] = 1
proc.update_state(BatchUpdate(**batch_kwargs))
for token in (1, 2):
    row = torch.zeros((1, 4), device="cpu")
    proc.apply(row)
    assert row.argmax().item() == token and torch.isfinite(row).sum().item() == 1
    history.append(token)
    proc.update_state(None)


class Client:
    async def generate(self, prompt, params, rid):
        assert isinstance(params, SamplingParams)
        assert params.output_kind == RequestOutputKind.FINAL_ONLY
        assert params.include_stop_str_in_output
        yield NS(
            finished=True,
            outputs=[
                NS(
                    text="<calc-final>2+3</calc-final>",
                    token_ids=[1, 2],
                    finish_reason="stop",
                )
            ],
        )

    async def abort(self, rid):
        raise AssertionError("unexpected abort")


segment = asyncio.run(NativeEngine(Client()).generate([0], 32, "qualification:0"))
assert segment.sampled_tokens == 2
from exactus.adapters.logits_adapter import _force_row

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    row = torch.zeros(4, dtype=dtype, device="cpu")
    _force_row(row, 1, 4)
    assert torch.isfinite(row).sum().item() == 1
    masked = torch.zeros(4, dtype=dtype, device="cpu")
    masked[1] = float("-inf")
    try:
        _force_row(masked, 1, 4)
    except Exception:
        assert masked[1].item() == float("-inf")
    else:
        raise AssertionError("masked token was reenabled")
print(
    json.dumps(
        {
            "candidate": __version__,
            "vllm": importlib.metadata.version("vllm"),
            "torch": torch.__version__,
            "device": "cpu",
            "actual_batch_update": True,
            "batch_signature": str(inspect.signature(BatchUpdate)),
            "batch_module": inspect.getfile(BatchUpdate),
            "endpoint_protocol": True,
            "actual_sampling_params": True,
            "cpu_dtypes": ["float32", "float16", "bfloat16"],
            "model_loaded": False,
            "gpu_inference": False,
            "service_changed": False,
        }
    )
)
