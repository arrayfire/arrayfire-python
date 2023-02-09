from typing import Any

from arrayfire.array_api import operators
from arrayfire.array_api.array_object import Array


class TestArithmeticOperators:
    def setup_method(self, method: Any) -> None:
        self.array1 = Array([1, 2, 3])
        self.array2 = Array([4, 5, 6])

    def test_add(self) -> None:
        res = operators.add(self.array1, self.array2)
        res_sum = self.array1 + self.array2
        assert res.to_list() == res_sum.to_list() == [5, 7, 9]

    def test_sub(self) -> None:
        res = operators.sub(self.array1, self.array2)
        res_sum = self.array1 - self.array2
        assert res.to_list() == res_sum.to_list() == [-3, -3, -3]
