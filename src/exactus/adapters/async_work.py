"""Keep admission ownership until non-cancellable thread work really ends."""

import asyncio
import contextvars
import functools


async def blocking(fn, *args, **kwargs):
    """On cancellation, drain the running thread before releasing its owner."""
    call = functools.partial(contextvars.copy_context().run, fn, *args, **kwargs)
    future = asyncio.get_running_loop().run_in_executor(None, call)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
            if future.cancelled():
                raise
            continue  # repeated cancellation must not abandon the running thread
        except Exception:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
        if cancelled:
            raise asyncio.CancelledError
        return result
