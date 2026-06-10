#define PY_SSIZE_T_CLEAN
#define ln_2 0.6931471805599453
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

// PyArray_* functions:
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

#include "math_utils.h"


typedef struct {
    double distance;
    double* paths;
} DTW;

typedef struct {
    size_t i;
    size_t j;
} Point;

typedef struct {
    Point* path;
    size_t len;
} WarpingPath;


// ============================================================================
// TIME-SERIES DISTANCE MEASURES
// ============================================================================

DTW dtw(const double* x, const double* y, size_t n_x, size_t n_y) {
    DTW result = {-1.0, NULL};
    double* D = malloc(n_x * n_y * sizeof(double));
    if (!D) return result;

    D[0] = _square(x[0] - y[0]);
    for (size_t i=1; i<n_x; i++) D[i * n_y] = D[(i-1) * n_y] + _square(x[i] - y[0]);
    for (size_t j=1; j<n_y; j++) D[j] = D[j-1] + _square(x[0] - y[j]);
    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            D[i * n_y + j] = _min3(D[(i-1) * n_y + j], D[i * n_y + (j-1)], D[(i-1) * n_y + (j-1)]) + _square(x[i] - y[j]);
        }
    }
    result.distance = D[(n_x-1) * n_y + (n_y-1)];
    result.paths = D;

    return result;
}

DTW lcss(const double* x, const double* y, size_t n_x, size_t n_y, double eps) {
    DTW result = {-1.0, NULL};
    double* D = malloc(n_x * n_y * sizeof(double));
    if (!D) return result;

    double cost_v = 0.0;
    double cost_h = 0.0;
    // D[0] = 0.0;
    // for (size_t i=1; i<n_x; i++) D[i * n_y] = D[(i-1) * n_y] + cost_v;
    // for (size_t j=1; j<n_y; j++) D[j] = D[j-1] + cost_h;
    D[0] = (_abs(x[0] - y[0]) <= eps) ? 1.0 : 0.0;
    for (size_t i=1; i<n_x; i++) {
        if (_abs(x[i] - y[0]) <= eps) D[i*n_y] = 1.0;
        else D[i*n_y] = D[(i-1) * n_y];
    }
    for (size_t j=1; j<n_y; j++) {
        if (_abs(x[0] - y[j]) <= eps) D[j] = 1.0;
        else D[j] = D[j-1];
    }
    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            // double cost_d = (_abs(x[i] - y[j]) <= eps ? 1.0 : 0.0);
            if (_abs(x[i] - y[j]) <= eps) 
                D[i * n_y + j] = D[(i-1) * n_y + (j-1)] + 1.0;
            else 
                D[i * n_y + j] = _max(D[(i-1) * n_y + (j)], D[(i) * n_y + (j-1)]);
            // D[i * n_y + j] = _min3(D[(i-1) * n_y + j] + cost_v, D[i * n_y + (j-1)] + cost_h, D[(i-1) * n_y + (j-1)] + cost_d);
        }
    }
    result.distance = 1.0 - D[(n_x-1) * n_y + (n_y-1)] / _min(n_x, n_y);
    result.paths = D;

    return result;
}

DTW edr(const double* x, const double* y, size_t n_x, size_t n_y, double eps) {
    DTW result = {-1.0, NULL};
    double* D = malloc(n_x * n_y * sizeof(double));
    if (!D) return result;

    double cost_d = (_abs(x[0] - y[0]) <= eps) ? 0.0 : 1.0;
    double cost_v = 1.0;
    double cost_h = 1.0;
    D[0] = _min3(0.0 + cost_d, 1.0 + cost_v, 1.0 + cost_h); 
    for (size_t i=1; i<n_x; i++) {
        double subcost = (_abs(x[i] - y[0]) <= eps) ? 0.0 : 1.0;
        D[i * n_y] = _min3((double)i + subcost, D[(i-1) * n_y] + cost_v, (double)(i+1) + cost_h);
    }

    for (size_t j=1; j<n_y; j++) {
        double subcost = (_abs(x[0] - y[j]) <= eps) ? 0.0 : 1.0;
        D[j] = _min3((double)j + subcost, (double)(j+1) + cost_v, D[j-1] + cost_h);
    }

    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            cost_d = (_abs(x[i] - y[j]) <= eps) ? 0.0 : 1.0;
            D[i * n_y + j] = _min3(D[(i-1) * n_y + (j-1)] + cost_d, D[(i-1) * n_y + j] + cost_v, D[i * n_y + (j-1)] + cost_h);
        }
    }

    result.distance = D[(n_x-1) * n_y + (n_y-1)] / _max((double)n_x, (double)n_y);
    result.paths = D;
    
    return result;
}

DTW swale(const double* x, const double* y, size_t n_x, size_t n_y, double eps, double p, double r) {
    DTW result = {-1.0, NULL};
    double* D = malloc(n_x * n_y * sizeof(double));
    if (!D) return result;

    double cost_v = p;
    double cost_h = p;
    D[0] = 0.0;
    for (size_t i=1; i<n_x; i++) D[i * n_y] = D[(i-1) * n_y] + cost_v;
    for (size_t j=1; j<n_y; j++) D[j] = D[j-1] + cost_h;
    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            double cost_d = (_abs(x[i] - y[j]) <= eps ? r : p);
            D[i * n_y + j] = _min3(D[(i-1) * n_y + j] + cost_v, D[i * n_y + (j-1)] + cost_h, D[(i-1) * n_y + (j-1)] + cost_d);
        }
    }
    result.distance = D[(n_x-1) * n_y + (n_y-1)];
    result.paths = D;

    return result;
}

DTW erp(const double* x, const double* y, size_t n_x, size_t n_y, double g) {
    DTW result = {-1.0, NULL};
    double* D = malloc(n_x * n_y * sizeof(double));
    if (!D) return result;

    D[0] = _square(x[0] - y[0]);
    for (size_t i=1; i<n_x; i++) D[i * n_y] = D[(i-1) * n_y] + _square(x[i] - g);
    for (size_t j=1; j<n_y; j++) D[j] = D[j-1] + _square(y[j] - g);
    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            D[i * n_y + j] = _min3(D[(i-1) * n_y + j] + _square(x[i] - g), 
                                   D[i * n_y + (j-1)] + _square(y[j] - g), 
                                   D[(i-1) * n_y + (j-1)] + _square(x[i] - y[j]));
        }
    }
    result.distance = D[(n_x-1) * n_y + (n_y-1)];
    result.paths = D;

    return result;
}

DTW msm(const double* x, const double* y, size_t n_x, size_t n_y, double c) {
    DTW result = {-1.0, NULL};
    double* D = malloc(n_x * n_y * sizeof(double));
    if (!D) return result;

    D[0] = _abs(x[0] - y[0]);
    for (size_t i=1; i<n_x; i++) {
        double cost_v = (((x[i-1] <= x[i]) && (x[i] <= y[0])) || ((x[i-1] >= x[i]) && (x[i] >= y[0])) ? c : c + _min(_abs(x[i] - x[i-1]), _abs(x[i] - y[0])));
        D[i * n_y] = D[(i-1) * n_y] + cost_v;
    }
    for (size_t j=1; j<n_y; j++) {
        double cost_h = (((y[j-1] <= y[j]) && (y[j] <= x[0])) || ((y[j-1] >= y[j]) && (y[j] >= x[0])) ? c : c + _min(_abs(y[j] - y[j-1]), _abs(y[j] - x[0])));
        D[j] = D[j-1] + cost_h;
    }
    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            double cost_v = (((x[i-1] <= x[i]) && (x[i] <= y[j])) || ((x[i-1] >= x[i]) && (x[i] >= y[j])) ? c : c + _min(_abs(x[i] - x[i-1]), _abs(x[i] - y[j])));
            double cost_h = (((y[j-1] <= y[j]) && (y[j] <= x[i])) || ((y[j-1] >= y[j]) && (y[j] >= x[i])) ? c : c + _min(_abs(y[j] - y[j-1]), _abs(y[j] - x[i])));
            double cost_d = _abs(x[i] - y[j]);
            D[i * n_y + j] = _min3(D[(i-1) * n_y + j] + cost_v, 
                                   D[i * n_y + (j-1)] + cost_h, 
                                   D[(i-1) * n_y + (j-1)] + cost_d);
        }
    }
    result.distance = D[(n_x-1) * n_y + (n_y-1)];
    result.paths = D;

    return result;
}

DTW twed(const double* x, const double* y, size_t n_x, size_t n_y, double nu, double lambda) {
    DTW result = {-1.0, NULL};
    double* D = malloc(n_x * n_y * sizeof(double));
    if (!D) return result;

    D[0] = 0.0;
    for (size_t i=1; i<n_x; i++) D[i * n_y] = D[(i-1) * n_y] + _square(x[i] - x[i-1]) + nu + lambda;
    for (size_t j=1; j<n_y; j++) D[j] = D[j-1] + _square(y[j] - y[j-1]) + nu + lambda;
    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            double cost_v = _square(x[i] - x[i-1]) + nu + lambda;
            double cost_h = _square(y[j] - y[j-1]) + nu + lambda;
            double cost_d = _square(x[i] - y[j]) + _square(x[i-1] - y[j-1]) + 2.0 * nu;
            D[i * n_y + j] = _min3(D[(i-1) * n_y + j] + cost_v, 
                                   D[i * n_y + (j-1)] + cost_h, 
                                   D[(i-1) * n_y + (j-1)] + cost_d);
        }
    }
    result.distance = D[(n_x-1) * n_y + (n_y-1)];
    result.paths = D;

    return result;
}

WarpingPath best_path(const double* D, size_t n_x, size_t n_y) {
    WarpingPath result = {NULL, 0};
    if (n_x == 0 || n_y == 0 || !D) return result;
    Point* path = malloc((n_x + n_y) * sizeof(Point));
    if (!path) return result;
    size_t idx = 0;
    size_t i = n_x - 1;
    size_t j = n_y - 1;

    path[idx] = (Point){i, j};
    idx++;

    while (i > 0 || j > 0) {
        if (i == 0) j--;
        else if (j == 0) i--;
        else {
            if (D[(i-1) * n_y + (j-1)] <= D[(i-1) * n_y + j] && D[(i-1) * n_y + (j-1)] <= D[(i * n_y + (j-1))]) {
                i--;
                j--;
            }
            else if (D[(i-1) * n_y + j] <= D[i * n_y + (j-1)]) i--;
            else j--;
        }
        path[idx] = (Point){i, j};
        idx++;
    }

    for (size_t i=0; i<idx/2; i++) {
        Point temp = path[i];
        path[i] = path[idx - i - 1];
        path[idx - i - 1] = temp;
    }

    result.path = path;
    result.len = idx;

    return result;
}


// ============================================================================
// PYTHON WRAPPER UTILITIES
// ============================================================================

static PyObject* py_dtw(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) != 2)
        return PyErr_Format(PyExc_RuntimeError, "Expected 2 arguments");

    PyObject* args0 = PyTuple_GetItem(args, 0);
    PyObject* args1 = PyTuple_GetItem(args, 1);
    if (!args0 || !args1) return NULL;

    if (!PyArray_Check(args0) || !PyArray_Check(args1))
        return PyErr_Format(PyExc_RuntimeError, "Expected numpy arrays");

    const PyArrayObject* _x = (const PyArrayObject*)args0;
    const PyArrayObject* _y = (const PyArrayObject*)args1;

    if (PyArray_NDIM(_x) != 1 || PyArray_NDIM(_y) != 1)
        return PyErr_Format(PyExc_TypeError, "Expected 1D numpy arrays");

    if (PyArray_TYPE(_x) != NPY_DOUBLE || PyArray_TYPE(_y) != NPY_DOUBLE)
        return PyErr_Format(PyExc_RuntimeError, "Expected numpy double-typed arrays");

    if (!PyArray_IS_C_CONTIGUOUS(_x) || !PyArray_IS_C_CONTIGUOUS(_y))
        return PyErr_Format(PyExc_RuntimeError, "Expected contiguous arrays");

    const double* x = PyArray_DATA(_x);
    const double* y = PyArray_DATA(_y);
    size_t n_x = PyArray_SIZE(_x);
    size_t n_y = PyArray_SIZE(_y);
    for (size_t i=0; i<n_x; i++) {
        if (isnan(x[i])) {
            return PyErr_Format(PyExc_ValueError, "Expected arrays with non-NA values");
        }
    }
    for (size_t j=0; j<n_y; j++) {
        if (isnan(y[j])) {
            return PyErr_Format(PyExc_ValueError, "Expected arrays with non-NA values");
        }
    }

    DTW result = dtw(x, y, n_x, n_y);
    if (!result.paths) 
        return PyErr_NoMemory();

    npy_intp dims[2] = {n_x, n_y};
    PyObject* py_paths = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if(!py_paths) {
        free(result.paths);
        return PyErr_NoMemory();
    }

    memcpy(PyArray_DATA((PyArrayObject*)py_paths), result.paths, n_x * n_y * sizeof(double));

    free(result.paths);

    PyObject* tuple_result = PyTuple_New(2);
    if (!tuple_result) {
        Py_DECREF(py_paths);
        return NULL;
    }
    PyTuple_SetItem(tuple_result, 0, PyFloat_FromDouble(result.distance));
    PyTuple_SetItem(tuple_result, 1, py_paths);
    
    return tuple_result;
}

static PyObject* py_pairwise_dtw(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 1)
        return PyErr_Format(PyExc_RuntimeError, "Expected 1 argument");

    PyObject* args0 = PyTuple_GetItem(args, 0);
    if (!args0) return NULL;

    if (!PyArray_Check(args0) || PyArray_NDIM((PyArrayObject*)args0) != 2)
        return PyErr_Format(PyExc_TypeError, "Expected a 2D numpy array");

    const PyArrayObject* _D = (const PyArrayObject*)args0;
    if (PyArray_TYPE(_D) != NPY_DOUBLE)
        return PyErr_Format(PyExc_RuntimeError, "Expected a 2D numpy double-typed array");

    if (!PyArray_IS_C_CONTIGUOUS(_D))
        return PyErr_Format(PyExc_RuntimeError, "Expected a 2D contiguous array");

    const double* D = PyArray_DATA(_D);
    size_t n = PyArray_DIM(_D, 0);
    size_t len = PyArray_DIM(_D, 1);
    size_t total = n * len;
    for (size_t i=0; i<total; i++) {
        if (isnan(D[i])) {
            return PyErr_Format(PyExc_ValueError, "Expected a 2D array with non-NA values");
        }
    }

    npy_intp dims[2] = {n, n};
    PyObject* py_dists = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!py_dists) return PyErr_NoMemory();
    double* dist_matrix = PyArray_DATA((PyArrayObject*)py_dists);

    int i;
    int j;
    
    // Py_BEGIN_ALLOW_THREADS

    // #pragma omp parallel for schedule(dynamic)
    for (i=0; i<(int)n; i++) {
        for (j=i; j<(int)n; j++) {
            const double* x = D + (i * len);
            const double* y = D + (j * len);
            DTW result = dtw(x, y, len, len);
            dist_matrix[i * n + j] = result.distance;
            dist_matrix[j * n + i] = result.distance;
            
            free(result.paths); 
        }
    }

    // Py_END_ALLOW_THREADS

    return py_dists;
}

#define PY_ELASTIC_3PARAMS(NAME) \
    static PyObject* py_##NAME(PyObject* self, PyObject* args) \
    { \
        PyObject *args0 = NULL; \
        PyObject *args1 = NULL; \
        \
        double param = (strcmp(#NAME, "msm") == 0) ? 1.0 : (strcmp(#NAME, "erp") == 0) ? 0.0 : 0.2; \
    \
        if (!PyArg_ParseTuple(args, "OO|d", &args0, &args1, &param)) \
            return NULL; \
        \
        if (strcmp(#NAME, "lcss") == 0 && param < 0.0) \
            return PyErr_Format(PyExc_ValueError, "Parameter 'eps' for LCSS must be non-negative"); \
        if (strcmp(#NAME, "edr") == 0 && param < 0.0) \
            return PyErr_Format(PyExc_ValueError, "Parameter 'eps' for EDR must be non-negative"); \
        if (strcmp(#NAME, "msm") == 0 && param < 0.0) \
            return PyErr_Format(PyExc_ValueError, "Parameter 'c' for MSM must be non-negative"); \
    \
        if (!PyArray_Check(args0) || !PyArray_Check(args1)) \
            return PyErr_Format(PyExc_RuntimeError, "Expected numpy arrays"); \
    \
        const PyArrayObject* _x = (const PyArrayObject*)args0; \
        const PyArrayObject* _y = (const PyArrayObject*)args1; \
        if (PyArray_NDIM(_x) != 1 || PyArray_NDIM(_y) != 1) \
            return PyErr_Format(PyExc_TypeError, "Expected 1D numpy arrays"); \
\
        if (PyArray_TYPE(_x) != NPY_DOUBLE || PyArray_TYPE(_y) != NPY_DOUBLE) \
            return PyErr_Format(PyExc_RuntimeError, "Expected numpy double-typed arrays"); \
    \
        if (!PyArray_IS_C_CONTIGUOUS(_x) || !PyArray_IS_C_CONTIGUOUS(_y)) \
            return PyErr_Format(PyExc_RuntimeError, "Expected contiguous arrays"); \
    \
        const double* x = PyArray_DATA(_x); \
        const double* y = PyArray_DATA(_y); \
        size_t n_x = PyArray_SIZE(_x); \
        size_t n_y = PyArray_SIZE(_y); \
        for (size_t i=0; i<n_x; i++) { \
            if (isnan(x[i])) { \
                return PyErr_Format(PyExc_ValueError, "Expected arrays with non-NA values"); \
            } \
        } \
        for (size_t j=0; j<n_y; j++) { \
            if (isnan(y[j])) { \
                return PyErr_Format(PyExc_ValueError, "Expected arrays with non-NA values"); \
            } \
        } \
    \
        DTW result = NAME(x, y, n_x, n_y, param); \
        if (!result.paths)  \
            return PyErr_NoMemory(); \
    \
        npy_intp dims[2] = {n_x, n_y}; \
        PyObject* py_paths = PyArray_SimpleNew(2, dims, NPY_DOUBLE); \
        if(!py_paths) { \
            free(result.paths); \
            return PyErr_NoMemory(); \
        } \
    \
        memcpy(PyArray_DATA((PyArrayObject*)py_paths), result.paths, n_x * n_y * sizeof(double)); \
    \
        free(result.paths); \
    \
        PyObject* tuple_result = PyTuple_New(2); \
        if (!tuple_result) { \
            Py_DECREF(py_paths); \
            return NULL; \
        } \
        PyTuple_SetItem(tuple_result, 0, PyFloat_FromDouble(result.distance)); \
        PyTuple_SetItem(tuple_result, 1, py_paths); \
        \
        return tuple_result; \
    }

PY_ELASTIC_3PARAMS(lcss)
PY_ELASTIC_3PARAMS(edr)
PY_ELASTIC_3PARAMS(erp)
PY_ELASTIC_3PARAMS(msm)

#if defined(_MSC_VER)
    #define OMP_PARALLEL_FOR __pragma(omp parallel for schedule(dynamic))
#else
    #define OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(dynamic)")
#endif

#define PY_ELASTIC_PAIRWISE_3PARAMS(NAME) \
    static PyObject* py_pairwise_##NAME(PyObject* self, PyObject* args) {  \
        PyObject *args0 = NULL; \
        \
        double param = (strcmp(#NAME, "msm") == 0) ? 1.0 : (strcmp(#NAME, "erp") == 0) ? 0.0 : 0.2; \
    \
        if (!PyArg_ParseTuple(args, "O|d", &args0, &param)) \
            return NULL; \
        \
        if (strcmp(#NAME, "lcss") == 0 && param < 0.0) \
            return PyErr_Format(PyExc_ValueError, "Parameter 'eps' for LCSS must be non-negative"); \
        if (strcmp(#NAME, "edr") == 0 && param < 0.0) \
            return PyErr_Format(PyExc_ValueError, "Parameter 'eps' for EDR must be non-negative"); \
        if (strcmp(#NAME, "msm") == 0 && param < 0.0) \
            return PyErr_Format(PyExc_ValueError, "Parameter 'c' for MSM must be non-negative"); \
    \
        if (!PyArray_Check(args0) || PyArray_NDIM((PyArrayObject*)args0) != 2) \
            return PyErr_Format(PyExc_TypeError, "Expected a 2D numpy array"); \
 \
        const PyArrayObject* _D = (const PyArrayObject*)args0; \
        if (PyArray_TYPE(_D) != NPY_DOUBLE) \
            return PyErr_Format(PyExc_RuntimeError, "Expected a 2D numpy double-typed array"); \
 \
        if (!PyArray_IS_C_CONTIGUOUS(_D)) \
            return PyErr_Format(PyExc_RuntimeError, "Expected a 2D contiguous array"); \
         \
        const double* D = PyArray_DATA(_D); \
        size_t n = PyArray_DIM(_D, 0); \
        size_t len = PyArray_DIM(_D, 1); \
        size_t total = n * len; \
        for (size_t i=0; i<total; i++) { \
            if (isnan(D[i])) { \
                return PyErr_Format(PyExc_ValueError, "Expected a 2D array with non-NA values"); \
            } \
        } \
 \
        npy_intp dims[2] = {n, n}; \
        PyObject* py_dists = PyArray_SimpleNew(2, dims, NPY_DOUBLE); \
        if (!py_dists) return PyErr_NoMemory(); \
        double* dist_matrix = PyArray_DATA((PyArrayObject*)py_dists); \
 \
        int i; \
        int j; \
    \
        Py_BEGIN_ALLOW_THREADS \
 \
        OMP_PARALLEL_FOR \
        for (i=0; i<(int)n; i++) { \
            for (j=i; j<(int)n; j++) { \
                const double* x = D + (i * len); \
                const double* y = D + (j * len); \
                DTW result = NAME(x, y, len, len, param); \
                dist_matrix[i * n + j] = result.distance; \
                dist_matrix[j * n + i] = result.distance; \
                 \
                free(result.paths);  \
            } \
        } \
 \
        Py_END_ALLOW_THREADS \
 \
        return py_dists; \
    }

PY_ELASTIC_PAIRWISE_3PARAMS(lcss)
PY_ELASTIC_PAIRWISE_3PARAMS(edr)
PY_ELASTIC_PAIRWISE_3PARAMS(erp)
PY_ELASTIC_PAIRWISE_3PARAMS(msm)

static PyObject* py_twed(PyObject* self, PyObject* args)
{
    PyObject *args0 = NULL;
    PyObject *args1 = NULL;
    double nu = 0.001;
    double lambda = 1.0;

    if (!PyArg_ParseTuple(args, "OO|dd", &args0, &args1, &nu, &lambda))
        return NULL;

    if (nu < 0.0 || lambda < 0.0)
        return PyErr_Format(PyExc_ValueError, "Parameters 'nu' and 'lambda' must be non-negative");
    
    if (!PyArray_Check(args0) || !PyArray_Check(args1))
        return PyErr_Format(PyExc_RuntimeError, "Expected numpy arrays");

    const PyArrayObject* _x = (const PyArrayObject*)args0;
    const PyArrayObject* _y = (const PyArrayObject*)args1;
    if (PyArray_NDIM(_x) != 1 || PyArray_NDIM(_y) != 1)
        return PyErr_Format(PyExc_TypeError, "Expected 1D numpy arrays");

    if (PyArray_TYPE(_x) != NPY_DOUBLE || PyArray_TYPE(_y) != NPY_DOUBLE)
        return PyErr_Format(PyExc_RuntimeError, "Expected numpy double-typed arrays");

    if (!PyArray_IS_C_CONTIGUOUS(_x) || !PyArray_IS_C_CONTIGUOUS(_y))
        return PyErr_Format(PyExc_RuntimeError, "Expected contiguous arrays");

    const double* x = PyArray_DATA(_x);
    const double* y = PyArray_DATA(_y);
    size_t n_x = PyArray_SIZE(_x);
    size_t n_y = PyArray_SIZE(_y);
    for (size_t i=0; i<n_x; i++) {
        if (isnan(x[i])) {
            return PyErr_Format(PyExc_ValueError, "Expected arrays with non-NA values");
        }
    }
    for (size_t j=0; j<n_y; j++) {
        if (isnan(y[j])) {
            return PyErr_Format(PyExc_ValueError, "Expected arrays with non-NA values");
        }
    }

    DTW result = twed(x, y, n_x, n_y, nu, lambda);
    if (!result.paths) 
        return PyErr_NoMemory();

    npy_intp dims[2] = {n_x, n_y};
    PyObject* py_paths = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if(!py_paths) {
        free(result.paths);
        return PyErr_NoMemory();
    }

    memcpy(PyArray_DATA((PyArrayObject*)py_paths), result.paths, n_x * n_y * sizeof(double));

    free(result.paths);

    PyObject* tuple_result = PyTuple_New(2);
    if (!tuple_result) {
        Py_DECREF(py_paths);
        return NULL;
    }
    PyTuple_SetItem(tuple_result, 0, PyFloat_FromDouble(result.distance));
    PyTuple_SetItem(tuple_result, 1, py_paths);
    
    return tuple_result;
}

static PyObject* py_pairwise_twed(PyObject* self, PyObject* args) {  
    PyObject *args0 = NULL;
    double nu = 0.001;
    double lambda = 1.0;

    if (!PyArg_ParseTuple(args, "O|dd", &args0, &nu, &lambda))
        return NULL;

    if (nu < 0.0 || lambda < 0.0)
        return PyErr_Format(PyExc_ValueError, "Parameters 'nu' and 'lambda' must be non-negative");

    if (!PyArray_Check(args0) || PyArray_NDIM((PyArrayObject*)args0) != 2) 
        return PyErr_Format(PyExc_TypeError, "Expected a 2D numpy array"); 

    const PyArrayObject* _D = (const PyArrayObject*)args0; 
    if (PyArray_TYPE(_D) != NPY_DOUBLE) 
        return PyErr_Format(PyExc_RuntimeError, "Expected a 2D numpy double-typed array"); 

    if (!PyArray_IS_C_CONTIGUOUS(_D)) 
        return PyErr_Format(PyExc_RuntimeError, "Expected a 2D contiguous array"); 
        
    const double* D = PyArray_DATA(_D); 
    size_t n = PyArray_DIM(_D, 0); 
    size_t len = PyArray_DIM(_D, 1); 
    size_t total = n * len;
    for (size_t i=0; i<total; i++) {
        if (isnan(D[i])) {
            return PyErr_Format(PyExc_ValueError, "Expected a 2D array with non-NA values");
        }
    }

    npy_intp dims[2] = {n, n}; 
    PyObject* py_dists = PyArray_SimpleNew(2, dims, NPY_DOUBLE); 
    if (!py_dists) return PyErr_NoMemory(); 
    double* dist_matrix = PyArray_DATA((PyArrayObject*)py_dists); 

    int i;
    int j;
    
    Py_BEGIN_ALLOW_THREADS 

    #pragma omp parallel for schedule(dynamic)
    for (i=0; i<(int)n; i++) { 
        for (j=i; j<(int)n; j++) { 
            const double* x = D + (i * len); 
            const double* y = D + (j * len); 
            DTW result = twed(x, y, len, len, nu, lambda); 
            dist_matrix[i * n + j] = result.distance; 
            dist_matrix[j * n + i] = result.distance; 
                
            free(result.paths);  
        } 
    } 

    Py_END_ALLOW_THREADS 

    return py_dists; 
}

static PyObject* py_swale(PyObject* self, PyObject* args)
{
    PyObject *args0 = NULL;
    PyObject *args1 = NULL;
    double eps = 0.2;
    double p = 1.0;
    double r = 0.0;

    if (!PyArg_ParseTuple(args, "OO|ddd", &args0, &args1, &eps, &p, &r))
        return NULL;

    if (eps < 0.0)
        return PyErr_Format(PyExc_ValueError, "Parameter 'eps' must be non-negative");
    if (p < 0.0 || r < 0.0)
        return PyErr_Format(PyExc_ValueError, "Parameters 'p' and 'r' must be non-negative");
    if (r > p)
        return PyErr_Format(PyExc_ValueError, "Parameter 'r' must be less than or equal to 'p'");

    if (!PyArray_Check(args0) || !PyArray_Check(args1))
        return PyErr_Format(PyExc_RuntimeError, "Expected numpy arrays");

    const PyArrayObject* _x = (const PyArrayObject*)args0;
    const PyArrayObject* _y = (const PyArrayObject*)args1;
    if (PyArray_NDIM(_x) != 1 || PyArray_NDIM(_y) != 1)
        return PyErr_Format(PyExc_TypeError, "Expected 1D numpy arrays");
        
    if (PyArray_TYPE(_x) != NPY_DOUBLE || PyArray_TYPE(_y) != NPY_DOUBLE)
        return PyErr_Format(PyExc_RuntimeError, "Expected numpy double-typed arrays");

    if (!PyArray_IS_C_CONTIGUOUS(_x) || !PyArray_IS_C_CONTIGUOUS(_y))
        return PyErr_Format(PyExc_RuntimeError, "Expected contiguous arrays");

    const double* x = PyArray_DATA(_x);
    const double* y = PyArray_DATA(_y);
    size_t n_x = PyArray_SIZE(_x);
    size_t n_y = PyArray_SIZE(_y);
    for (size_t i=0; i<n_x; i++) {
        if (isnan(x[i])) {
            return PyErr_Format(PyExc_ValueError, "Expected arrays with non-NA values");
        }
    }
    for (size_t j=0; j<n_y; j++) {
        if (isnan(y[j])) {
            return PyErr_Format(PyExc_ValueError, "Expected arrays with non-NA values");
        }
    }
    
    DTW result = swale(x, y, n_x, n_y, eps, p, r);
    if (!result.paths) 
        return PyErr_NoMemory();

    npy_intp dims[2] = {n_x, n_y};
    PyObject* py_paths = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if(!py_paths) {
        free(result.paths);
        return PyErr_NoMemory();
    }

    memcpy(PyArray_DATA((PyArrayObject*)py_paths), result.paths, n_x * n_y * sizeof(double));

    free(result.paths);

    PyObject* tuple_result = PyTuple_New(2);
    if (!tuple_result) {
        Py_DECREF(py_paths);
        return NULL;
    }
    PyTuple_SetItem(tuple_result, 0, PyFloat_FromDouble(result.distance));
    PyTuple_SetItem(tuple_result, 1, py_paths);
    
    return tuple_result;
}

static PyObject* py_pairwise_swale(PyObject* self, PyObject* args) {  
    PyObject *args0 = NULL;
    double eps = 0.2;
    double p = 1.0;
    double r = 0.0;

    if (!PyArg_ParseTuple(args, "O|ddd", &args0, &eps, &p, &r))
        return NULL;

    if (eps < 0.0)
        return PyErr_Format(PyExc_ValueError, "Parameter 'eps' must be non-negative");
    if (p < 0.0 || r < 0.0)
        return PyErr_Format(PyExc_ValueError, "Parameters 'p' and 'r' must be non-negative");
    if (r > p)
        return PyErr_Format(PyExc_ValueError, "Parameter 'r' must be less than or equal to 'p'");

    if (!PyArray_Check(args0) || PyArray_NDIM((PyArrayObject*)args0) != 2) 
        return PyErr_Format(PyExc_TypeError, "Expected a 2D numpy array"); 

    const PyArrayObject* _D = (const PyArrayObject*)args0; 
    if (PyArray_TYPE(_D) != NPY_DOUBLE) 
        return PyErr_Format(PyExc_RuntimeError, "Expected a 2D numpy double-typed array"); 

    if (!PyArray_IS_C_CONTIGUOUS(_D)) 
        return PyErr_Format(PyExc_RuntimeError, "Expected a 2D contiguous array"); 
        
    const double* D = PyArray_DATA(_D); 
    size_t n = PyArray_DIM(_D, 0); 
    size_t len = PyArray_DIM(_D, 1); 
    size_t total = n * len;
    for (size_t i=0; i<total; i++) {
        if (isnan(D[i])) {
            return PyErr_Format(PyExc_ValueError, "Expected a 2D array with non-NA values");
        }
    }

    npy_intp dims[2] = {n, n}; 
    PyObject* py_dists = PyArray_SimpleNew(2, dims, NPY_DOUBLE); 
    if (!py_dists) return PyErr_NoMemory(); 
    double* dist_matrix = PyArray_DATA((PyArrayObject*)py_dists); 

    int i;
    int j;
    
    Py_BEGIN_ALLOW_THREADS 

    #pragma omp parallel for schedule(dynamic)
    for (i=0; i<(int)n; i++) { 
        for (j=i; j<(int)n; j++) { 
            const double* x = D + (i * len); 
            const double* y = D + (j * len); 
            DTW result = swale(x, y, len, len, eps, p, r); 
            dist_matrix[i * n + j] = result.distance; 
            dist_matrix[j * n + i] = result.distance; 
                
            free(result.paths);  
        } 
    } 

    Py_END_ALLOW_THREADS 

    return py_dists; 
}

static PyObject* py_best_path(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) != 1)
        return PyErr_Format(PyExc_RuntimeError, "Expected 1 argument");

    PyObject* args0 = PyTuple_GetItem(args, 0);
    if (!args0) return NULL;

    if (!PyArray_Check(args0) || PyArray_NDIM((PyArrayObject*)args0) != 2)
        return PyErr_Format(PyExc_TypeError, "Expected a 2D numpy array");

    const PyArrayObject* _D = (const PyArrayObject*)args0;
    if (PyArray_TYPE(_D) != NPY_DOUBLE)
        return PyErr_Format(PyExc_RuntimeError, "Expected a 2D numpy double-typed array");

    if (!PyArray_IS_C_CONTIGUOUS(_D))
        return PyErr_Format(PyExc_RuntimeError, "Expected a 2D contiguous array");

    const double* D = PyArray_DATA(_D);
    size_t n_x = PyArray_DIM(_D, 0);
    size_t n_y = PyArray_DIM(_D, 1);
    size_t total = n_x * n_y;
    for (size_t i=0; i<total; i++) {
        if (isnan(D[i])) {
            return PyErr_Format(PyExc_ValueError, "Expected a 2D array with non-NA values");
        }
    }

    WarpingPath result = best_path(D, n_x, n_y);
    if (!result.path) 
        return PyErr_NoMemory();

    npy_intp dims[2] = {result.len, 2};
    PyObject* py_path = PyArray_SimpleNew(2, dims, NPY_INT64);
    int64_t* path = PyArray_DATA((PyArrayObject*)py_path);
    for (size_t i=0; i<result.len; i++) {
        path[i*2] = (int64_t)result.path[i].i;
        path[i*2+1] = (int64_t)result.path[i].j;
    }
    
    free(result.path);
    
    return py_path;
}


// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

static PyMethodDef elastic_cmodule_methods[] = {
    {"dtw", py_dtw, METH_VARARGS, "Calculate Dynamic Time Warping"},
    {"lcss", py_lcss, METH_VARARGS, "Calculate Longest Common Subsequence"},
    {"edr", py_edr, METH_VARARGS, "Calculate Edit Distance on Real Sequences"},
    {"erp", py_erp, METH_VARARGS, "Calculate Edit Distance with Real Penalty"},
    {"msm", py_msm, METH_VARARGS, "Calculate Move-Split-Merge"},
    {"twed", py_twed, METH_VARARGS, "Calculate Time Warp Edit Distance"},
    {"swale", py_swale, METH_VARARGS, "Calculate Sequence Weighted Alignment"},
    {"pairwise_dtw", py_pairwise_dtw, METH_VARARGS, "Calculate pairwise Dynamic Time Warping"},
    {"pairwise_lcss", py_pairwise_lcss, METH_VARARGS, "Calculate pairwise Longest Common Subsequence"},
    {"pairwise_edr", py_pairwise_edr, METH_VARARGS, "Calculate pairwise Edit Distance on Real Sequences"},
    {"pairwise_erp", py_pairwise_erp, METH_VARARGS, "Calculate pairwise Edit Distance with Real Penalty"},
    {"pairwise_msm", py_pairwise_msm, METH_VARARGS, "Calculate pairwise Move-Split-Merge"},
    {"pairwise_twed", py_pairwise_twed, METH_VARARGS, "Calculate pairwise Time Warp Edit Distance"},
    {"pairwise_swale", py_pairwise_swale, METH_VARARGS, "Calculate pairwise Sequence Weighted Alignment"},
    {"best_path", py_best_path, METH_VARARGS, "Extract best path from a cost matrix"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef elastic_cmodule_module = {
    PyModuleDef_HEAD_INIT,
    "elastic_cmodule", 
    "Elastic Time-Series Distance Measures including "
    "Dynamic Time Warping (DTW), "
    "Threshold-based and Metric Elastic Measures.",
    -1,
    elastic_cmodule_methods
};

PyMODINIT_FUNC PyInit_elastic() {
    PyObject* mod = PyModule_Create(&elastic_cmodule_module);
    if (!mod) return NULL;
    import_array(); 
    return mod;
}

