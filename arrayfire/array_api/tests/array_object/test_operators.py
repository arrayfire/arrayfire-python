import operator
from typing import Any, List, Union

import pytest

from arrayfire.array_api.array_object import Array


# HACK replace for e.g. abs(x1-x2) < 1e-6 ~ https://davidamos.dev/the-right-way-to-compare-floats-in-python/
def _round(list_: List[Union[int, float]], symbols: int = 4) -> List[Union[int, float]]:
    return [round(x, symbols) for x in list_]


def pytest_generate_tests(metafunc: Any) -> None:
    if "array_origin" in metafunc.fixturenames:
        metafunc.parametrize("array_origin", [
            [1, 2, 3],
            # [4.2, 7.5, 5.41]  # FIXME too big difference between python pow and af backend
        ])
    if "op_origin" in metafunc.fixturenames:
        metafunc.parametrize("op_origin", [
            "add",  # __add__, __iadd__, __radd__
            "sub",  # __sub__, __isub__, __rsub__
            "mul",  # __mul__, __imul__, __rmul__
            "truediv",  # __truediv__, __itruediv__, __rtruediv__
            # "floordiv",  # __floordiv__, __ifloordiv__, __rfloordiv__  # TODO
            "mod",  # __mod__, __imod__, __rmod__
            "pow"  # __pow__, __ipow__, __rpow__,
        ])
    if "operand" in metafunc.fixturenames:
        metafunc.parametrize("operand", [
            2,
            1.5,
            [9, 9, 9],
        ])
    if "false_operand" in metafunc.fixturenames:
        metafunc.parametrize("false_operand", [
            (1, 2, 3),
            ("2"),
            {2.34, 523.2},
            "15"
        ])


def test_arithmetic_operators(
        array_origin: List[Union[int, float]], op_origin: str,
        operand: Union[int, float, List[Union[int, float]]]) -> None:
    op = getattr(operator, op_origin)
    iop = getattr(operator, "i" + op_origin)

    if isinstance(operand, list):
        ref = [op(x, y) for x, y in zip(array_origin, operand)]
        rref = [op(y, x) for x, y in zip(array_origin, operand)]
        operand = Array(operand)  # type: ignore[assignment]
    else:
        ref = [op(x, operand) for x in array_origin]
        rref = [op(operand, x) for x in array_origin]

    array = Array(array_origin)

    res = op(array, operand)
    ires = iop(array, operand)
    rres = op(operand, array)

    assert _round(res.to_list()) == _round(ires.to_list()) == _round(ref)
    assert _round(rres.to_list()) == _round(rref)

    assert res.dtype == ires.dtype == rres.dtype
    assert res.ndim == ires.ndim == rres.ndim
    assert res.size == ires.size == ires.size
    assert res.shape == ires.shape == rres.shape
    assert len(res) == len(ires) == len(rres)


def test_arithmetic_operators_expected_to_raise_error(
        array_origin: List[Union[int, float]], op_origin: str, false_operand: Any) -> None:
    array = Array(array_origin)
    op = getattr(operator, op_origin)
    with pytest.raises(TypeError):
        op(array, false_operand)
