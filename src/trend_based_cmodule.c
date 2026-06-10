#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <omp.h>
#include <math.h>

// PyArray_* functions:
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

#include "math_utils.h"


// ============================================================================
// TIME-SERIES DISTANCE MEASURES
// ============================================================================

double edt(const double* x, const double* y, size_t n, double lambda) {
    double dist = _square(x[0] - y[0]);
    int flag = _sgn(x[0] - y[0]);
    int current_flag;
    for (size_t i=1; i<n; i++) {
        current_flag = _sgn(x[i] - y[i]);
        if (current_flag == flag || current_flag == 0 || flag == 0) dist += _square(x[i] - y[i]);
        else dist += (_square(x[i] - y[i]) * lambda);
        if (current_flag != 0) flag = current_flag;
    }
    
    return _sqrt(dist / n);
}

double edtd(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=1; i<n; i++) {
        double deriv_x = x[i] - x[i-1];
        double deriv_y = y[i] - y[i-1];
        double diff_i = _abs(x[i] - y[i]);
        double diff_i_1 = _abs(x[i-1] - y[i-1]);
        dist += (_square((diff_i + diff_i_1) / 2.0) * (1.0 + _abs(deriv_x - deriv_y)));
    }
    
    return _sqrt(dist / (n-1));
}


// ============================================================================
// PYTHON WRAPPER UTILITIES
// ============================================================================

static PyObject* py_edt(PyObject* self, PyObject* args) {
    PyObject *args0 = NULL;
    PyObject *args1 = NULL;
    double lambda = 1.1;

    if (!PyArg_ParseTuple(args, "OO|d", &args0, &args1, &lambda))
        return NULL;

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

    if (PyArray_SIZE(_x) != PyArray_SIZE(_y)) 
        return PyErr_Format(PyExc_ValueError, "Expected same size arrays"); 

    const double* x = PyArray_DATA(_x); 
    const double* y = PyArray_DATA(_y); 
    size_t n = PyArray_SIZE(_x);

    for (size_t i=0; i<n; i++) {
        if (isnan(x[i]) || isnan(y[i])) {
            return PyErr_Format(PyExc_ValueError, "Expected arrays with non-NA values");
        }
    }

    return PyFloat_FromDouble(edt(x, y, n, lambda));
}

static PyObject* py_pairwise_edt(PyObject* self, PyObject* args) {  
    PyObject *args0 = NULL;
    double lambda = 1.1;

    if (!PyArg_ParseTuple(args, "O|d", &args0, &lambda))
        return NULL;

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
    double result;

    Py_BEGIN_ALLOW_THREADS 

    #pragma omp parallel for schedule(dynamic) private(j, result)
    for (i=0; i<(int)n; i++) { 
        for (j=i; j<(int)n; j++) { 
            const double* x = D + (i * len); 
            const double* y = D + (j * len); 
            result = edt(x, y, len, lambda); 
            dist_matrix[i * n + j] = result; 
            dist_matrix[j * n + i] = result; 
        } 
    } 

    Py_END_ALLOW_THREADS 

    return py_dists; 
}

static PyObject* py_edtd(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2)
        return PyErr_Format(PyExc_RuntimeError, "Expected 2 arguments (array, array)");
    
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

    if (PyArray_SIZE(_x) != PyArray_SIZE(_y)) 
        return PyErr_Format(PyExc_ValueError, "Expected same size arrays"); 

    const double* x = PyArray_DATA(_x); 
    const double* y = PyArray_DATA(_y);
    size_t n = PyArray_SIZE(_x); 

    for (size_t i=0; i<n; i++) {
        if (isnan(x[i]) || isnan(y[i])) {
            return PyErr_Format(PyExc_ValueError, "Expected arrays with non-NA values");
        }
    }

    return PyFloat_FromDouble(edtd(x, y, n));
}

static PyObject* py_pairwise_edtd(PyObject* self, PyObject* args) {
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
    double result;

    Py_BEGIN_ALLOW_THREADS

    #pragma omp parallel for schedule(dynamic) private(j, result)
    for (i=0; i<(int)n; i++) {
        for (j=i; j<(int)n; j++) {
            const double* x = D + (i * len);
            const double* y = D + (j * len);
            result = edtd(x, y, len);
            dist_matrix[i * n + j] = result;
            dist_matrix[j * n + i] = result;
        }
    }

    Py_END_ALLOW_THREADS

    return py_dists;
}


// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

static PyMethodDef trend_based_cmodule_methods[] = {
    {"edt", py_edt, METH_VARARGS, "Calculate Trend-Based Euklidean Distance"},
    {"edtd", py_edtd, METH_VARARGS, "Calculate Derivative-Based Euklidean Distance"},
    {"pairwise_edt", py_pairwise_edt, METH_VARARGS, "Calculate pairwise Trend-Based Euklidean Distance"},
    {"pairwise_edtd", py_pairwise_edtd, METH_VARARGS, "Calculate pairwise Derivative-Based Euklidean Distance"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef trend_based_cmodule_module = {
    PyModuleDef_HEAD_INIT,
    "trend_based_cmodule", 
    "Trend-Based Time-Series Distance Measures including "
    "Trend-Based Euklidean Distance, "
    "Derivative-Based Euklidean Distance.",
    -1,
    trend_based_cmodule_methods
};

PyMODINIT_FUNC PyInit_trend_based() {
    PyObject* mod = PyModule_Create(&trend_based_cmodule_module);
    if (!mod) return NULL;
    import_array(); 
    return mod;
}
