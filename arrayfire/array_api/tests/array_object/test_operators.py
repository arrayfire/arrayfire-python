import operator
from typing import Any, Callable, List, Union

import pytest

from arrayfire.array_api.array_object import Array
from arrayfire.array_api.dtypes import bool as af_bool

Operator = Callable[..., Any]

arithmetic_operators = [
    [operator.add, operator.iadd],
    [operator.sub, operator.isub],
    [operator.mul, operator.imul],
    [operator.truediv, operator.itruediv],
    [operator.mod, operator.imod],
    [operator.pow, operator.ipow]
]

comparison_operators = [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne]


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
        metafunc.parametrize("arithmetic_operator", arithmetic_operators)
    if "comparison_operator" in metafunc.fixturenames:
        metafunc.parametrize("comparison_operator", comparison_operators)
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
        array_origin: List[Union[int, float]], arithmetic_operator: List[Operator],
        operand: Union[int, float, List[Union[int, float]]]) -> None:
    op = arithmetic_operator[0]
    iop = arithmetic_operator[1]

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
        array_origin: List[Union[int, float]], arithmetic_operator: List[Operator], false_operand: Any) -> None:
    array = Array(array_origin)

    with pytest.raises(TypeError):
        op = arithmetic_operator[0]
        op(array, false_operand)

    # BUG string type false operand never raises an error
    # with pytest.raises(TypeError):
    #     op = arithmetic_operator[0]
    #     op(false_operand, array)

    with pytest.raises(TypeError):
        op = arithmetic_operator[1]
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
