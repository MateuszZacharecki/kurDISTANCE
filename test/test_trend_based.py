import pytest
import numpy as np
from numpy.testing import assert_allclose
from typing import Tuple

import kurDISTANCE.trend_based as tb


@pytest.fixture
def time_series_pair() -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(123)
    x = np.random.uniform(size=(10,))
    y = np.random.uniform(size=(10,))
    return x, y


@pytest.fixture
def time_series_dataset() -> np.ndarray:
    np.random.seed(123)
    return np.random.uniform(size=(5, 10))


@pytest.fixture
def non_contiguous_array() -> np.ndarray:
    np.random.seed(123)
    return np.random.uniform(size=(10, 5))[:, 0]


@pytest.fixture
def contiguous_array() -> np.ndarray:
    np.random.seed(123)
    return np.random.uniform(size=(10,))


@pytest.fixture
def non_contiguous_dataset() -> np.ndarray:
    np.random.seed(123)
    return np.random.uniform(size=(5, 10)).T


def test_edt_returns_zero_for_identical_arrays(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair
    y = x.copy()

    # When
    distance = tb.edt(x, y)

    # Then
    assert distance == pytest.approx(0.0, abs=1e-7)


def test_edt_calculates_valid_distance(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    lambd = 1.5

    # When
    distance = tb.edt(x, y, lambd)

    # Then
    assert isinstance(distance, float)
    assert distance >= 0.0


def test_edt_calculates_valid_distance_with_default_args(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair

    # When
    distance = tb.edt(x, y)

    # Then
    assert isinstance(distance, float)
    assert distance >= 0.0


def test_edt_raises_type_error_when_too_few_arguments(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair

    # When / Then
    with pytest.raises(TypeError):
        tb.edt(x)


def test_edt_raises_type_error_when_too_many_arguments(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    arg1 = 1.5
    arg2 = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        tb.edt(x, y, arg1, arg2)


def test_edt_raises_type_error_when_third_argument_is_not_a_double(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    lambd = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        tb.edt(x, y, lambd)


def test_edt_raises_type_error_for_non_1d_arrays() -> None:
    # Given
    array_1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    array_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    array_0d = np.array(5.0, dtype=np.float64)

    # When / Then
    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        tb.edt(array_2d, array_1d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        tb.edt(array_1d, array_2d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        tb.edt(array_2d, array_2d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        tb.edt(array_0d, array_1d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        tb.edt(array_1d, array_0d)


def test_edt_raises_runtime_error_for_invalid_dtype(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    x_int = x.astype(np.int32)

    # When / Then
    with pytest.raises(RuntimeError, match="Expected numpy double-typed arrays"):
        tb.edt(x_int, y)


def test_edt_raises_runtime_error_when_first_array_is_not_contiguous(
    non_contiguous_array: np.ndarray, contiguous_array: np.ndarray
) -> None:
    # Given
    x = non_contiguous_array
    y = contiguous_array

    # When / Then
    with pytest.raises(RuntimeError, match="Expected contiguous arrays"):
        tb.edt(x, y)


def test_edt_raises_runtime_error_when_second_array_is_not_contiguous(
    contiguous_array: np.ndarray, non_contiguous_array: np.ndarray
) -> None:
    # Given
    x = contiguous_array
    y = non_contiguous_array

    # When / Then
    with pytest.raises(RuntimeError, match="Expected contiguous arrays"):
        tb.edt(x, y)


def test_edt_raises_value_error_for_mismatched_sizes(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    y_short = y[:-1]

    # When / Then
    with pytest.raises(ValueError, match="Expected same size arrays"):
        tb.edt(x, y_short)


def test_pairwise_edt_returns_symmetric_matrix_with_zero_diagonal(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    lambd = 1.5

    # When
    dist_matrix = tb.pairwise_edt(dataset, lambd)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-7)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-7)


def test_pairwise_edt_returns_symmetric_matrix_with_zero_diagonal_with_default_args(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset 

    # When
    dist_matrix = tb.pairwise_edt(dataset)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-7)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-7)


def test_pairwise_edt_raises_type_error_when_too_few_arguments(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    _ = time_series_dataset

    # When / Then
    with pytest.raises(TypeError):
        tb.pairwise_edt()


def test_pairwise_edt_raises_type_error_when_too_many_arguments(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    arg1 = 1.5
    arg2 = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        tb.pairwise_edt(dataset, arg1, arg2)


def test_pairwise_edt_raises_type_error_when_second_argument_is_not_a_double(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    lambd = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        tb.pairwise_edt(dataset, lambd)


def test_pairwise_edt_raises_type_error_for_1d_array(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair

    # When / Then
    with pytest.raises(TypeError, match="Expected a 2D numpy array"):
        tb.pairwise_edt(x)


def test_pairwise_edt_raises_runtime_error_for_invalid_dtype(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    dataset_int = dataset.astype(np.int32)

    # When / Then
    with pytest.raises(RuntimeError, match="Expected a 2D numpy double-typed array"):
        tb.pairwise_edt(dataset_int)


def test_pairwise_edt_raises_runtime_error_when_array_is_not_contiguous(
    non_contiguous_dataset: np.ndarray
) -> None:
    # Given
    dataset = non_contiguous_dataset

    # When / Then
    with pytest.raises(RuntimeError, match="Expected a 2D contiguous array"):
        tb.pairwise_edt(dataset)


def test_edtd_returns_zero_for_identical_arrays(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair
    y = x.copy()

    # When
    distance = tb.edtd(x, y)

    # Then
    assert distance == pytest.approx(0.0, abs=1e-7)


def test_edtd_calculates_valid_distance(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair

    # When
    distance = tb.edtd(x, y)

    # Then
    assert isinstance(distance, float)
    assert distance >= 0.0


def test_edtd_raises_runtime_error_when_too_few_arguments(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair

    # When / Then
    with pytest.raises(RuntimeError, match="Expected 2 arguments (array, array)"):
        tb.edtd(x)


def test_edtd_raises_runtime_error_when_too_many_arguments(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    arg1 = 1.5
    arg2 = "pupa"

    # When / Then
    with pytest.raises(RuntimeError, match="Expected 2 arguments (array, array)"):
        tb.edtd(x, y, arg1, arg2)


def test_edtd_raises_type_error_for_non_1d_arrays() -> None:
    # Given
    array_1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    array_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    array_0d = np.array(5.0, dtype=np.float64)

    # When / Then
    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        tb.edtd(array_2d, array_1d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        tb.edtd(array_1d, array_2d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        tb.edtd(array_2d, array_2d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        tb.edtd(array_0d, array_1d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        tb.edtd(array_1d, array_0d)


def test_edtd_raises_runtime_error_for_invalid_dtype(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    x_int = x.astype(np.int32)

    # When / Then
    with pytest.raises(RuntimeError, match="Expected numpy double-typed arrays"):
        tb.edtd(x_int, y)


def test_edtd_raises_runtime_error_when_first_array_is_not_contiguous(
    non_contiguous_array: np.ndarray, contiguous_array: np.ndarray
) -> None:
    # Given
    x = non_contiguous_array
    y = contiguous_array

    # When / Then
    with pytest.raises(RuntimeError, match="Expected contiguous arrays"):
        tb.edtd(x, y)


def test_edtd_raises_runtime_error_when_second_array_is_not_contiguous(
    contiguous_array: np.ndarray, non_contiguous_array: np.ndarray
) -> None:
    # Given
    x = contiguous_array
    y = non_contiguous_array

    # When / Then
    with pytest.raises(RuntimeError, match="Expected contiguous arrays"):
        tb.edtd(x, y)


def test_edtd_raises_value_error_for_mismatched_sizes(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    y_short = y[:-1]

    # When / Then
    with pytest.raises(ValueError, match="Expected same size arrays"):
        tb.edtd(x, y_short)


def test_pairwise_edtd_returns_symmetric_matrix_with_zero_diagonal(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset

    # When
    dist_matrix = tb.pairwise_edtd(dataset)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-7)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-7)


def test_pairwise_edtd_raises_runtime_error_when_too_few_arguments(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    _ = time_series_dataset

    # When / Then
    with pytest.raises(RuntimeError, match="Expected 1 argument"):
        tb.pairwise_edtd()


def test_pairwise_edtd_raises_runtime_error_when_too_many_arguments(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    arg1 = 1.5
    arg2 = "pupa"

    # When / Then
    with pytest.raises(RuntimeError, match="Expected 1 argument"):
        tb.pairwise_edtd(dataset, arg1, arg2)


def test_pairwise_edtd_raises_type_error_for_1d_array(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair

    # When / Then
    with pytest.raises(TypeError, match="Expected a 2D numpy array"):
        tb.pairwise_edtd(x)


def test_pairwise_edtd_raises_runtime_error_for_invalid_dtype(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    dataset_int = dataset.astype(np.int32)

    # When / Then
    with pytest.raises(RuntimeError, match="Expected a 2D numpy double-typed array"):
        tb.pairwise_edtd(dataset_int)


def test_pairwise_edtd_raises_runtime_error_when_array_is_not_contiguous(
    non_contiguous_dataset: np.ndarray
) -> None:
    # Given
    dataset = non_contiguous_dataset

    # When / Then
    with pytest.raises(RuntimeError, match="Expected a 2D contiguous array"):
        tb.pairwise_edtd(dataset)
