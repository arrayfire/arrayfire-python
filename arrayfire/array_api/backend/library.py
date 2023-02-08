from typing import Any

from . import AFArray, Pointer, backend, safe_call


@safe_call
def af_add(out: Pointer, lhs: AFArray, rhs: AFArray, batch: bool) -> Any:
    return backend.af_add(out, lhs, rhs, batch)
