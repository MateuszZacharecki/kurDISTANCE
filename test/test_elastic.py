import pytest
import numpy as np
from numpy.testing import assert_allclose
from typing import Tuple

import kurDISTANCE.elastic as elastic


# METRICS = [elastic.lcss, elastic.edr, elastic.erp, elastic.msm]
# METRICS_PAIRWISE = [elastic.pairwise_lcss, elastic.pairwise_edr, elastic.pairwise_erp, elastic.pairwise_msm]

ELASTIC_FUNCS = []
for name in dir(elastic): 
    if callable(getattr(elastic, name)) and not name.startswith("_"):
        ELASTIC_FUNCS.append(getattr(elastic, name)) 

METRICS = [f for f in ELASTIC_FUNCS if not f.__name__.startswith("pairwise") and f.__name__ != "best_path"]
METRICS_PAIRWISE = [f for f in ELASTIC_FUNCS if f.__name__.startswith("pairwise")]


@pytest.fixture
def time_series_pair() -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(123)
    x = np.random.uniform(size=(10,))
    y = np.random.uniform(size=(10,))
    return x, y


@pytest.fixture
def time_series_mismatched_pair() -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(123)
    x = np.random.uniform(size=(12,))
    y = np.random.uniform(size=(8,))
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

    # When
    distance, paths = func(x, y)

    # Then
    if func.__name__ == "twed":
        assert distance >= 0.0
    else:
        assert distance == pytest.approx(0.0, abs=1e-6)
    assert isinstance(paths, np.ndarray)
    assert paths.shape == (len(x), len(x))


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_calculates_valid_distance(func,
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair

    # When
    distance, paths = func(x, y)

    # Then
    assert isinstance(distance, float)
    assert distance >= 0.0
    assert paths.shape == (len(x), len(y))


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_calculates_valid_distance_1arg(func,
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    if "dtw" in func.__name__:
        pytest.skip(f"Skipping {func.__name__}: does not take an extra parameter.")
    # Given
    x, y = time_series_mismatched_pair
    param = 1.5

    # When
    distance, paths = func(x, y, param)

    # Then
    assert isinstance(distance, float)
    assert distance >= 0.0
    assert paths.shape == (len(x), len(y))


def test_twed_calculates_valid_distance_2args(
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    param1 = 1.5
    param2 = 2

    # When
    distance, paths = elastic.twed(x, y, param1, param2)

    # Then
    assert isinstance(distance, float)
    assert distance >= 0.0
    assert paths.shape == (len(x), len(y))


def test_swale_calculates_valid_distance_2args(
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    param1 = 1.5
    param2 = 2

    # When
    distance, paths = elastic.swale(x, y, param1, param2)

    # Then
    assert isinstance(distance, float)
    assert distance >= 0.0
    assert paths.shape == (len(x), len(y))


def test_swale_calculates_valid_distance_3args(
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    param1 = 2
    param2 = 1.5
    param3 = 3

    # When
    distance, paths = elastic.swale(x, y, param1, param2, param3)

    # Then
    assert isinstance(distance, float)
    assert distance >= 0.0
    assert paths.shape == (len(x), len(y))


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_type_error_when_too_few_arguments(func,
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_mismatched_pair

    # When / Then
    if func.__name__ == 'dtw':
        with pytest.raises(RuntimeError, match="Expected 2 arguments"):
            func(x)
    else:
        with pytest.raises(TypeError):
            func(x)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_type_error_when_too_many_arguments(func,
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    arg1 = 1.5
    arg2 = "pupa"
    arg3 = 3
    arg4 = 4

    # When / Then
    if func.__name__ == 'dtw':
        with pytest.raises(RuntimeError, match="Expected 2 arguments"):
            func(x, y, arg1, arg2, arg3, arg4)
    else:
        with pytest.raises(TypeError):
            func(x, y, arg1, arg2, arg3, arg4)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_type_error_when_argument_is_not_a_double_1arg(func,
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    if "dtw" in func.__name__:
        pytest.skip(f"Skipping {func.__name__}: does not take an extra parameter.")
    # Given
    x, y = time_series_mismatched_pair
    param = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        func(x, y, param)


def test_twed_raises_type_error_when_argument_is_not_a_double_2args(
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    param1 = 1.5
    param2 = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        elastic.twed(x, y, param1, param2)


def test_swale_raises_type_error_when_argument_is_not_a_double_2args(
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    param1 = 1.5
    param2 = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        elastic.swale(x, y, param1, param2)


def test_swale_raises_type_error_when_argument_is_not_a_double_3args(
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    param1 = 1.5
    param2 = 2
    param3 = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        elastic.swale(x, y, param1, param2, param3)


def test_lcss_raises_type_error_when_third_argument_is_negative(
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    eps = -0.1

    with pytest.raises(ValueError, match="Parameter 'eps' for LCSS must be non-negative"):
        elastic.lcss(x, y, eps)


def test_edr_raises_type_error_when_third_argument_is_negative(
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    eps = -0.1

    with pytest.raises(ValueError, match="Parameter 'eps' for EDR must be non-negative"):
        elastic.edr(x, y, eps)


def test_msm_raises_type_error_when_third_argument_is_negative(
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    c = -0.1

    with pytest.raises(ValueError, match="Parameter 'c' for MSM must be non-negative"):
        elastic.msm(x, y, c)


@pytest.mark.parametrize("nu, lambd, error", [
    (-1.0, 0.5, "Parameters 'nu' and 'lambda' must be non-negative"),
    (1.0, -0.5, "Parameters 'nu' and 'lambda' must be non-negative"),
    (-1.0, -0.5, "Parameters 'nu' and 'lambda' must be non-negative"),
], ids=["nu<0.0", "lambda<0.0", "nu_lambda<0.0"])
def test_twed_parameter_validation_raises_value_error(
    time_series_mismatched_pair, nu: float, lambd: float, error: str
) -> None:
    # Given
    x, y = time_series_mismatched_pair

    # When / Then
    with pytest.raises(ValueError, match=error):
        elastic.twed(x, y, nu, lambd)


@pytest.mark.parametrize("eps, p, r, error", [
    (-0.1, 1.0, 0.5, "Parameter 'eps' must be non-negative"),
    (0.1, -1.0, 0.5, "Parameters 'p' and 'r' must be non-negative"),
    (0.1, 1.0, -0.5, "Parameters 'p' and 'r' must be non-negative"),
    (0.1, -1.0, -0.5, "Parameters 'p' and 'r' must be non-negative"),
    (0.1, 1.0, 1.5, "Reward parameter 'r' must be less than or equal to penalty 'p'"),
], ids=["eps<0.0", "p<0.0", "r<0.0", "p_r<0.0", "r>p"])
def test_swale_parameter_validation_raises_value_error(
    time_series_mismatched_pair, eps: float, p: float, r: float, error: str
) -> None:
    # Given
    x, y = time_series_mismatched_pair

    # When / Then
    with pytest.raises(ValueError, match=error):
        elastic.swale(x, y, eps, p, r)


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
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
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
def test_func_raises_value_error_for_nan_in_first_array(func,
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
    x_with_nan = x.copy()
    x_with_nan[3] = np.nan

    # When / Then
    with pytest.raises(ValueError, match="Expected arrays with non-NA values"):
        func(x_with_nan, y)


@pytest.mark.parametrize("func", METRICS, ids=lambda f: f.__name__)
def test_func_raises_value_error_for_nan_in_second_array(func,
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, y = time_series_mismatched_pair
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

    # When
    dist_matrix = pairwise_func(dataset)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-6)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-6)


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_returns_symmetric_matrix_with_zero_diagonal_1arg(pairwise_func,
    time_series_dataset: np.ndarray
) -> None:
    if "pairwise_dtw" in pairwise_func.__name__:
        pytest.skip(f"Skipping {pairwise_func.__name__}: does not take an extra parameter.")

    # Given
    dataset = time_series_dataset
    param = 1.5

    # When
    dist_matrix = pairwise_func(dataset, param)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-6)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-6)


def test_pairwise_twed_returns_symmetric_matrix_with_zero_diagonal_2args(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    param1 = 1.5
    param2 = 2

    # When
    dist_matrix = elastic.pairwise_twed(dataset, param1, param2)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-6)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-6)


def test_pairwise_swale_returns_symmetric_matrix_with_zero_diagonal_2args(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    param1 = 1.5
    param2 = 2

    # When
    dist_matrix = elastic.pairwise_swale(dataset, param1, param2)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-6)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-6)


def test_pairwise_swale_returns_symmetric_matrix_with_zero_diagonal_3args(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    param1 = 2
    param2 = 1.5
    param3 = 3

    # When
    dist_matrix = elastic.pairwise_swale(dataset, param1, param2, param3)

    # Then
    assert dist_matrix.shape == (5, 5)
    assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-6)
    assert_allclose(dist_matrix, dist_matrix.T, atol=1e-6)


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_raises_type_error_when_too_few_arguments(pairwise_func,
    time_series_dataset: np.ndarray
) -> None:
    # Given
    _ = time_series_dataset

    # When / Then
    if pairwise_func.__name__ == 'pairwise_dtw':
        with pytest.raises(RuntimeError, match="Expected 2 arguments"):
            pairwise_func()
    else:
        with pytest.raises(TypeError):
            pairwise_func()


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_raises_type_error_when_too_many_arguments(pairwise_func,
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    arg1 = 1.5
    arg2 = "pupa"
    arg3 = 3
    arg4 = 4

    # When / Then
    if pairwise_func.__name__ == 'dtw':
        with pytest.raises(RuntimeError, match="Expected 2 arguments"):
            pairwise_func(dataset, arg1, arg2, arg3, arg4)
    else:
        with pytest.raises(TypeError):
            pairwise_func(dataset, arg1, arg2, arg3, arg4)


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_raises_type_error_when_argument_is_not_a_double_1arg(pairwise_func,
    time_series_dataset: np.ndarray
) -> None:
    if "pairwise_dtw" in pairwise_func.__name__:
        pytest.skip(f"Skipping {pairwise_func.__name__}: does not take an extra parameter.")
    # Given
    dataset = time_series_dataset
    param = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        pairwise_func(dataset, param)


def test_pairwise_twed_raises_type_error_when_argument_is_not_a_double_2args(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    param1 = 1.5
    param2 = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        elastic.pairwise_twed(dataset, param1, param2)


def test_pairwise_swale_raises_type_error_when_argument_is_not_a_double_2args(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    param1 = 1.5
    param2 = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        elastic.pairwise_swale(dataset, param1, param2)


def test_pairwise_swale_raises_type_error_when_argument_is_not_a_double_3args(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    param1 = 1.5
    param2 = 2
    param3 = "pupa"

    # When / Then
    with pytest.raises(TypeError):
        elastic.pairwise_swale(dataset, param1, param2, param3)


def test_pairwise_lcss_raises_type_error_when_third_argument_is_negative(
    time_series_dataset: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    dataset = time_series_dataset
    eps = -0.1

    with pytest.raises(ValueError, match="Parameter 'eps' for LCSS must be non-negative"):
        elastic.pairwise_lcss(dataset, eps)


def test_pairwise_edr_raises_type_error_when_third_argument_is_negative(
    time_series_dataset: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    dataset = time_series_dataset
    eps = -0.1

    with pytest.raises(ValueError, match="Parameter 'eps' for EDR must be non-negative"):
        elastic.pairwise_edr(dataset, eps)


def test_pairwise_msm_raises_type_error_when_third_argument_is_negative(
    time_series_dataset: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    dataset = time_series_dataset
    c = -0.1

    with pytest.raises(ValueError, match="Parameter 'c' for MSM must be non-negative"):
        elastic.pairwise_msm(dataset, c)


@pytest.mark.parametrize("nu, lambd, error", [
    (-1.0, 0.5, "Parameters 'nu' and 'lambda' must be non-negative"),
    (1.0, -0.5, "Parameters 'nu' and 'lambda' must be non-negative"),
    (-1.0, -0.5, "Parameters 'nu' and 'lambda' must be non-negative"),
], ids=["nu<0.0", "lambda<0.0", "nu_lambda<0.0"])
def test_pairwise_twed_parameter_validation_raises_value_error(
    time_series_dataset, nu: float, lambd: float, error: str
) -> None:
    # Given
    dataset = time_series_dataset

    # When / Then
    with pytest.raises(ValueError, match=error):
        elastic.pairwise_twed(dataset, nu, lambd)


@pytest.mark.parametrize("eps, p, r, error", [
    (-0.1, 1.0, 0.5, "Parameter 'eps' must be non-negative"),
    (0.1, -1.0, 0.5, "Parameters 'p' and 'r' must be non-negative"),
    (0.1, 1.0, -0.5, "Parameters 'p' and 'r' must be non-negative"),
    (0.1, -1.0, -0.5, "Parameters 'p' and 'r' must be non-negative"),
    (0.1, 1.0, 1.5, "Reward parameter 'r' must be less than or equal to penalty 'p'"),
], ids=["eps<0.0", "p<0.0", "r<0.0", "p_r<0.0", "r>p"])
def test_pairwise_swale_parameter_validation_raises_value_error(
    time_series_dataset, eps: float, p: float, r: float, error: str
) -> None:
    # Given
    dataset = time_series_dataset

    # When / Then
    with pytest.raises(ValueError, match=error):
        elastic.pairwise_swale(dataset, eps, p, r)


@pytest.mark.parametrize("pairwise_func", METRICS_PAIRWISE, ids=lambda f: f.__name__)
def test_pairwise_func_raises_type_error_for_1d_array(pairwise_func,
    time_series_mismatched_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_mismatched_pair

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


def test_best_path_returns_same_shaped_matrix(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    n_x, n_y = dataset.shape

    # When
    path = elastic.best_path(dataset)

    # Then
    assert path.shape[1] == 2 
    assert max(n_x, n_y) <= path.shape[0] <= n_x + n_y - 1


def test_best_path_raises_type_error_when_too_few_arguments(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    _ = time_series_dataset

    # When / Then
    with pytest.raises(RuntimeError, match="Expected 1 argument"):
        elastic.best_path()


def test_best_path_raises_type_error_when_too_many_arguments(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    arg1 = 1.5
    arg2 = "pupa"

    # When / Then
    with pytest.raises(RuntimeError, match="Expected 1 argument"):
        elastic.best_path(dataset, arg1, arg2)


def test_best_path_raises_type_error_for_1d_array(
    time_series_pair: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Given
    x, _ = time_series_pair

    # When / Then
    with pytest.raises(TypeError, match="Expected a 2D numpy array"):
        elastic.best_path(x)


def test_best_path_raises_runtime_error_for_invalid_dtype(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset = time_series_dataset
    dataset_int = dataset.astype(np.int32)

    # When / Then
    with pytest.raises(RuntimeError, match="Expected a 2D numpy double-typed array"):
        elastic.best_path(dataset_int)


def test_best_path_raises_runtime_error_when_array_is_not_contiguous(
    non_contiguous_dataset: np.ndarray
) -> None:
    # Given
    dataset = non_contiguous_dataset

    # When / Then
    with pytest.raises(RuntimeError, match="Expected a 2D contiguous array"):
        elastic.best_path(dataset)


def test_best_path_raises_value_error_for_nan_in_dataset(
    time_series_dataset: np.ndarray
) -> None:
    # Given
    dataset_with_nan = time_series_dataset.copy()
    dataset_with_nan[2, 4] = np.nan

    # When / Then
    with pytest.raises(ValueError, match="Expected a 2D array with non-NA values"):
        elastic.best_path(dataset_with_nan)


