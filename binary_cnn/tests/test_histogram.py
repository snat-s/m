import numpy as np
import pytest

from binary_cnn.data import extract_byte_windows
from binary_cnn.histogram import byte_pair_histogram


def test_adjacent_pairs_have_expected_coordinates_and_weights() -> None:
    histogram = byte_pair_histogram(bytes((1, 2, 1, 2)))

    assert histogram.shape == (256, 256)
    assert histogram[1, 2] == pytest.approx(2 / 3)
    assert histogram[2, 1] == pytest.approx(1 / 3)
    assert np.count_nonzero(histogram) == 2
    assert histogram.sum() == pytest.approx(1.0)


def test_offset_pairs_and_short_input() -> None:
    histogram = byte_pair_histogram(bytes((4, 8, 15, 16)), offset=2)
    assert histogram[4, 15] == pytest.approx(0.5)
    assert histogram[8, 16] == pytest.approx(0.5)
    assert byte_pair_histogram(b"x").sum() == 0


def test_offset_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        byte_pair_histogram(b"abc", offset=0)


def test_byte_windows_preserve_edges_and_pad_in_the_expected_directions() -> None:
    windows = extract_byte_windows(b" \x01\x02\n", window_size=4).reshape(3, 4)

    assert windows[0].tolist() == [1, 2, 10, 256]
    assert windows[1].tolist() == [32, 1, 2, 10]
    assert windows[2].tolist() == [256, 32, 1, 2]
