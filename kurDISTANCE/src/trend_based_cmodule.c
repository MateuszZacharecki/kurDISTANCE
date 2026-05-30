#define PY_SSIZE_T_CLEAN
#include <Python.h>

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
        dist += _square(x[i] - y[i]);
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
        dist += (_square((diff_i + diff_i_1) / 2) * (1.0 + _abs(deriv_x - deriv_y)));
    }
    
    return _sqrt(dist / (n-1));
}

