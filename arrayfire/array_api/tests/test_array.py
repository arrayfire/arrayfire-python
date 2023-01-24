from arrayfire.array_api import Array, float32


def test_empty_array() -> None:
    array = Array()

    assert array.dtype == float32
    assert array.ndim == 0
    assert array.size == 0
    assert array.shape == ()
    assert len(array) == 0
