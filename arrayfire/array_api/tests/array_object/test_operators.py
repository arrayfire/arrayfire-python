import operator
from typing import Any, Callable, List, Union

import pytest

from arrayfire.array_api.array_object import Array
from arrayfire.array_api.dtypes import bool as af_bool

Operator = Callable[[Union[int, float, Array], Union[int, float, Array]], Array]


def _round(list_: List[Union[int, float]], symbols: int = 4) -> List[Union[int, float]]:
    # HACK replace for e.g. abs(x1-x2) < 1e-6 ~ https://davidamos.dev/the-right-way-to-compare-floats-in-python/
    return [round(x, symbols) for x in list_]


def pytest_generate_tests(metafunc: Any) -> None:
    if "array_origin" in metafunc.fixturenames:
        metafunc.parametrize("array_origin", [
            [1, 2, 3],
            # [4.2, 7.5, 5.41]  # FIXME too big difference between python pow and af backend
        ])
    if "arithmetic_operator" in metafunc.fixturenames:
        metafunc.parametrize("arithmetic_operator", [
            "add",  # __add__, __iadd__, __radd__
            "sub",  # __sub__, __isub__, __rsub__
            "mul",  # __mul__, __imul__, __rmul__
            "truediv",  # __truediv__, __itruediv__, __rtruediv__
            # "floordiv",  # __floordiv__, __ifloordiv__, __rfloordiv__  # TODO
            "mod",  # __mod__, __imod__, __rmod__
            "pow",  # __pow__, __ipow__, __rpow__,
        ])
    if "array_operator" in metafunc.fixturenames:
        metafunc.parametrize("array_operator", [
            operator.matmul,
            operator.imatmul
        ])
    if "comparison_operator" in metafunc.fixturenames:
        metafunc.parametrize("comparison_operator", [
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            operator.eq,
            operator.ne
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
        array_origin: List[Union[int, float]], arithmetic_operator: str,
        operand: Union[int, float, List[Union[int, float]]]) -> None:
    op = getattr(operator, arithmetic_operator)
    iop = getattr(operator, "i" + arithmetic_operator)

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
        array_origin: List[Union[int, float]], arithmetic_operator: str, false_operand: Any) -> None:
    array = Array(array_origin)
    op = getattr(operator, arithmetic_operator)
    with pytest.raises(TypeError):
        op(array, false_operand)


def test_comparison_operators(
        array_origin: List[Union[int, float]], comparison_operator: Operator,
        operand: Union[int, float, List[Union[int, float]]]) -> None:
    if isinstance(operand, list):
        ref = [comparison_operator(x, y) for x, y in zip(array_origin, operand)]
        operand = Array(operand)  # type: ignore[assignment]
    else:
        ref = [comparison_operator(x, operand) for x in array_origin]

    array = Array(array_origin)
    res = comparison_operator(array, operand)  # type: ignore[arg-type]

    assert res.to_list() == ref
    assert res.dtype == af_bool


def test_comparison_operators_expected_to_raise_error(
        array_origin: List[Union[int, float]], comparison_operator: Operator, false_operand: Any) -> None:
    array = Array(array_origin)

    with pytest.raises(TypeError):
        comparison_operator(array, false_operand)
