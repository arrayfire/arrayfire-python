from arrayfire.array_api.array_object import Array

# TODO add more tests for different dtypes


def test_array_getitem() -> None:
    array = Array([1, 2, 3, 4, 5])

    int_item = array[2]
    assert array.dtype == int_item.dtype
    assert int_item.scalar() == 3


def test_scalar() -> None:
    array = Array([1, 2, 3])
    assert array[1].scalar() == 2


def test_scalar_is_empty() -> None:
    array = Array()
    assert array.scalar() is None


def test_array_to_list() -> None:
    array = Array([1, 2, 3])
    assert array.to_list() == [1, 2, 3]


def test_array_to_list_is_empty() -> None:
    array = Array()
    assert array.to_list() == []
