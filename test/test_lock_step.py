import re
import pytest
import numpy as np
from numpy.testing import assert_allclose
from typing import Tuple

import kurDISTANCE.lock_step as ls


LS_FUNCS = []
for name in dir(ls): 
    if callable(getattr(ls, name)) and not name.startswith("_"):
        LS_FUNCS.append(getattr(ls, name)) 

METRICS = [f for f in LS_FUNCS if not f.__name__.startswith("pairwise") and f.__name__ != "minkowski"]
METRICS_PAIRWISE = [f for f in LS_FUNCS if f.__name__.startswith("pairwise") and f.__name__ != "pairwise_minkowski"]


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


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_returns_zero_for_identical_arrays(func,
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair
    y = x.copy()

    if func.__name__ in ["inner_product", "harmonic_mean", "kumar_hassebrook", "cosine", "fidelity", "bhattacharyya", 'motyka']:
        pytest.skip(f"Skipping identity test: {func.__name__} is a similarity metric.")

    # When
    distance = func(x, y)

    # Then
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_motyka_returns_half_for_identical_arrays(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair
    y = x.copy()

    # When
    distance = ls.motyka(x, y)

    # Then
    assert distance == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_calculates_valid_distance(func,
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair

    # When
    distance = func(x, y)

    # Then
    assert isinstance(distance, float)
    assert distance >= 0.0


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_type_error_when_too_few_arguments(func,
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair

    # When / Then
    with pytest.raises(RuntimeError, match=re.escape("Expected 2 arguments (array, array)")):
        func(x)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_type_error_when_too_many_arguments(func,
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    arg1 = 1.5
    arg2 = "pupa"

    # When / Then
    with pytest.raises(RuntimeError, match=re.escape("Expected 2 arguments (array, array)")):
        func(x, y, arg1, arg2)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_funcs_raises_type_error_for_non_1d_arrays(func) -> None:
    # Given
    array_1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    array_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    array_0d = np.array(5.0, dtype=np.float64)

    # When / Then
    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        func(array_2d, array_1d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        func(array_1d, array_2d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        func(array_2d, array_2d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        func(array_0d, array_1d)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        func(array_1d, array_0d)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_runtime_error_for_invalid_dtype(func,
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    x_int = x.astype(np.int32)

    # When / Then
    with pytest.raises(RuntimeError, match="Expected numpy double-typed arrays"):
        func(x_int, y)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_runtime_error_when_first_array_is_not_contiguous(func,
    non_contiguous_array: np.ndarray, contiguous_array: np.ndarray
) -> None:
    # Given
    x = non_contiguous_array
    y = contiguous_array

    # When / Then
    with pytest.raises(RuntimeError, match="Expected contiguous arrays"):
        func(x, y)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_runtime_error_when_second_array_is_not_contiguous(func,
    contiguous_array: np.ndarray, non_contiguous_array: np.ndarray
) -> None:
    # Given
    x = contiguous_array
    y = non_contiguous_array

    # When / Then
    with pytest.raises(RuntimeError, match="Expected contiguous arrays"):
        func(x, y)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_value_error_for_mismatched_sizes(func,
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    y_short = y[:-1]

    # When / Then
    with pytest.raises(ValueError, match="Expected same size arrays"):
        func(x, y_short)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_value_error_for_nan_in_first_array(func,
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    x_with_nan = x.copy()
    x_with_nan[3] = np.nan

    # When / Then
    with pytest.raises(ValueError, match="Expected arrays with non-NA values"):
        func(x_with_nan, y)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_value_error_for_nan_in_second_array(func,
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    y_with_nan = y.copy()
    y_with_nan[5] = np.nan

    # When / Then
    with pytest.raises(ValueError, match="Expected arrays with non-NA values"):
        func(x, y_with_nan)


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_returns_symmetric_matrix_with_zero_diagonal(pairwise_func,
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset

    if pairwise_func.__name__ in ["pairwise_inner_product", "pairwise_harmonic_mean", "pairwise_kumar_hassebrook", "pairwise_cosine", "pairwise_fidelity", "pairwise_bhattacharyya", "pairwise_motyka"]:
        pytest.skip(f"Skipping diagonal test: {pairwise_func.__name__} is a similarity metric.")

    # When
    dist_matrix = pairwise_func(dataset)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-6)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-6)


def test_pairwise_motyka_returns_symmetric_matrix_with_half_diagonal(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset

    # When
    dist_matrix = ls.pairwise_motyka(dataset)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.5, atol=1e-6)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-6)


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_raises_type_error_when_too_few_arguments(pairwise_func,
    time_series_dataset: np.ndarray
) -> None:
    # Given
    _ = time_series_dataset

    # When / Then
    with pytest.raises(RuntimeError, match="Expected 1 argument"):
        pairwise_func()


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_raises_type_error_when_too_many_arguments(pairwise_func,
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    arg1 = 1.5
    arg2 = "pupa"

    # When / Then
    with pytest.raises(RuntimeError, match="Expected 1 argument"):
        pairwise_func(dataset, arg1, arg2)


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_raises_type_error_for_1d_array(pairwise_func,
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair

    # When / Then
    with pytest.raises(TypeError, match="Expected a 2D numpy array"):
        pairwise_func(x)


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_raises_runtime_error_for_invalid_dtype(pairwise_func,
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    dataset_int = dataset.astype(np.int32)

    # When / Then
    with pytest.raises(RuntimeError, match="Expected a 2D numpy double-typed array"):
        pairwise_func(dataset_int)


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_raises_runtime_error_when_array_is_not_contiguous(pairwise_func,
    non_contiguous_dataset: np.ndarray
) -> None:
    # Given
    dataset = non_contiguous_dataset

    # When / Then
    with pytest.raises(RuntimeError, match="Expected a 2D contiguous array"):
        pairwise_func(dataset)


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_raises_value_error_for_nan_in_dataset(pairwise_func, 
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset_with_nan = time_series_dataset.copy()
    dataset_with_nan[2, 4] = np.nan

    # When / Then
    with pytest.raises(ValueError, match="Expected a 2D array with non-NA values"):
        pairwise_func(dataset_with_nan)


def test_minkowski_returns_zero_for_identical_arrays(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair
    y = x.copy()
    p = 3

    # When
    distance = ls.minkowski(x, y, p)

    # Then
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_minkowski_calculates_valid_distance(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    p = 3

    # When
    distance = ls.minkowski(x, y, p)

    # Then
    assert isinstance(distance, float)
    assert distance >= 0.0


def test_minkowski_1d_mathematical_equivalence(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair

    # When
    minkowski_p1 = ls.minkowski(x, y, 1.0)
    minkowski_p2 = ls.minkowski(x, y, 2.0)
    
    # Reference metrics from your module
    manhattan = ls.manhattan(x, y)
    euclidean = ls.euclidean(x, y)

    # Then
    assert minkowski_p1 == pytest.approx(manhattan, abs=1e-6)
    assert minkowski_p2 == pytest.approx(euclidean, abs=1e-6)


def test_minkowski_raises_type_error_when_too_few_arguments(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair

    # When / Then
    with pytest.raises(RuntimeError, match=re.escape("Expected 3 arguments (array, array, p)")):
        ls.minkowski(x, y)


def test_minkowski_raises_type_error_when_too_many_arguments(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    arg1 = 3
    arg2 = "pupa"

    # When / Then
    with pytest.raises(RuntimeError, match=re.escape("Expected 3 arguments (array, array, p)")):
        ls.minkowski(x, y, arg1, arg2)


def test_minkowski_raises_type_error_when_third_argument_is_not_a_double(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    p = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        ls.minkowski(x, y, p)


def test_minkowski_raises_type_error_when_second_argument_is_not_an_int(
    time_series_pair: np.ndarray
) -> None:
    # Given
    x, y = time_series_pair
    p = 1.5

    # When / Then
    with pytest.raises(TypeError, match="Parameter 'p' must be an integer.*got 'float'"):
        ls.minkowski(x, y, p)


def test_minkowski_raises_type_error_when_third_argument_is_less_than_one(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    p = 0

    # When / Then
    with pytest.raises(ValueError, match="Parameter 'p' must be greater than or equal to 1"):
        ls.minkowski(x, y, p)


def test_minkowski_raises_type_error_for_non_1d_arrays() -> None:
    # Given
    array_1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    array_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    array_0d = np.array(5.0, dtype=np.float64)
    p = 3

    # When / Then
    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        ls.minkowski(array_2d, array_1d, p)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        ls.minkowski(array_1d, array_2d, p)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        ls.minkowski(array_2d, array_2d, p)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        ls.minkowski(array_0d, array_1d, p)

    with pytest.raises(TypeError, match="Expected 1D numpy arrays"):
        ls.minkowski(array_1d, array_0d, p)


def test_minkowski_raises_runtime_error_for_invalid_dtype(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    p = 3
    x_int = x.astype(np.int32)

    # When / Then
    with pytest.raises(RuntimeError, match="Expected numpy double-typed arrays"):
        ls.minkowski(x_int, y, p)


def test_minkowski_raises_runtime_error_when_first_array_is_not_contiguous(
    non_contiguous_array: np.ndarray, contiguous_array: np.ndarray
) -> None:
    # Given
    x = non_contiguous_array
    y = contiguous_array
    p = 3

    # When / Then
    with pytest.raises(RuntimeError, match="Expected contiguous arrays"):
        ls.minkowski(x, y, p)


def test_minkowski_raises_runtime_error_when_second_array_is_not_contiguous(
    contiguous_array: np.ndarray, non_contiguous_array: np.ndarray
) -> None:
    # Given
    x = contiguous_array
    y = non_contiguous_array
    p = 3

    # When / Then
    with pytest.raises(RuntimeError, match="Expected contiguous arrays"):
        ls.minkowski(x, y, p)


def test_minkowski_raises_value_error_for_mismatched_sizes(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    p = 3
    y_short = y[:-1]

    # When / Then
    with pytest.raises(ValueError, match="Expected same size arrays"):
        ls.minkowski(x, y_short, p)


def test_minkowski_raises_value_error_for_nan_in_first_array(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    x_with_nan = x.copy()
    x_with_nan[3] = np.nan
    p = 3

    # When / Then
    with pytest.raises(ValueError, match="Expected arrays with non-NA values"):
        ls.minkowski(x_with_nan, y, p)


def test_minkowski_raises_value_error_for_nan_in_second_array(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_pair
    y_with_nan = y.copy()
    y_with_nan[5] = np.nan
    p = 3

    # When / Then
    with pytest.raises(ValueError, match="Expected arrays with non-NA values"):
        ls.minkowski(x, y_with_nan, p)


def test_pairwise_minkowski_returns_symmetric_matrix_with_zero_diagonal(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    p = 3

    # When
    dist_matrix = ls.pairwise_minkowski(dataset, p)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-6)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-6)


def test_pairwise_minkowski_raises_type_error_when_too_few_arguments(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset

    # When / Then
    with pytest.raises(RuntimeError, match="Expected 2 arguments"):
        ls.pairwise_minkowski(dataset)


def test_pairwise_minkowski_raises_type_error_when_too_many_arguments(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    arg1 = 3
    arg2 = "pupa"

    # When / Then
    with pytest.raises(RuntimeError, match="Expected 2 arguments"):
        ls.pairwise_minkowski(dataset, arg1, arg2)


def test_pairwise_minkowski_raises_type_error_when_second_argument_is_not_a_number(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    p = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        ls.pairwise_minkowski(dataset, p)


def test_pairwise_minkowski_raises_type_error_when_second_argument_is_not_an_int(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    p = 1.5

    # When / Then
    with pytest.raises(TypeError, match="Parameter 'p' must be an integer.*got 'float'"):
        ls.pairwise_minkowski(dataset, p)


def test_pairwise_minkowski_raises_type_error_when_second_argument_is_less_than_one(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    p = 0

    # When / Then
    with pytest.raises(ValueError, match="Parameter 'p' must be greater than or equal to 1"):
        ls.pairwise_minkowski(dataset, p)


def test_pairwise_minkowski_raises_type_error_for_1d_array(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair
    p = 3

    # When / Then
    with pytest.raises(TypeError, match="Expected a 2D numpy array"):
        ls.pairwise_minkowski(x, p)


def test_pairwise_minkowski_raises_runtime_error_for_invalid_dtype(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    p = 3
    dataset_int = dataset.astype(np.int32)

    # When / Then
    with pytest.raises(RuntimeError, match="Expected a 2D numpy double-typed array"):
        ls.pairwise_minkowski(dataset_int, p)


def test_pairwise_minkowski_raises_runtime_error_when_array_is_not_contiguous(
    non_contiguous_dataset: np.ndarray
) -> None:
    # Given
    dataset = non_contiguous_dataset
    p = 3

    # When / Then
    with pytest.raises(RuntimeError, match="Expected a 2D contiguous array"):
        ls.pairwise_minkowski(dataset, p)


def test_pairwise_minkowski_mathematical_equivalence(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset

    # When
    pairwise_minkowski_p1 = ls.pairwise_minkowski(dataset, 1)
    pairwise_minkowski_p2 = ls.pairwise_minkowski(dataset, 2)
    
    pairwise_manhattan = ls.pairwise_manhattan(dataset)
    pairwise_euclidean = ls.pairwise_euclidean(dataset)

    # Then
    assert_allclose(pairwise_minkowski_p1, pairwise_manhattan, atol=1e-6)
    assert_allclose(pairwise_minkowski_p2, pairwise_euclidean, atol=1e-6)


def test_pairwise_minkowski_raises_value_error_for_nan_in_dataset(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset_with_nan = time_series_dataset.copy()
    dataset_with_nan[2, 4] = np.nan
    p = 3

    # When / Then
    with pytest.raises(ValueError, match="Expected a 2D array with non-NA values"):
        ls.pairwise_minkowski(dataset_with_nan, p)
