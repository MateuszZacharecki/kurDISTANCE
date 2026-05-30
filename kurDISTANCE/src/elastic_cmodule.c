#define PY_SSIZE_T_CLEAN
#define ln_2 0.6931471805599453
#include <Python.h>
#include <stdlib.h>

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
    if (!D) return -1.0;

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
    if (!D) return -1.0;

    double cost_v = 0.0;
    double cost_h = 0.0;
    D[0] = 0.0;
    for (size_t i=1; i<n_x; i++) D[i * n_y] = D[(i-1) * n_y] + cost_v;
    for (size_t j=1; j<n_y; j++) D[j] = D[j-1] + cost_h;
    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            double cost_d = (_abs(x[i] - y[j]) <= eps ? 1.0 : 0.0);
            D[i * n_y + j] = _min3(D[(i-1) * n_y + j] + cost_v, D[i * n_y + (j-1)] + cost_h, D[(i-1) * n_y + (j-1)] + cost_d);
        }
    }
    result.distance = 1 - D[(n_x-1) * n_y + (n_y-1)] / _min(n_x, n_y);
    result.paths = D;

    return result;
}

DTW edr(const double* x, const double* y, size_t n_x, size_t n_y, double eps) {
    DTW result = {-1.0, NULL};
    double* D = malloc(n_x * n_y * sizeof(double));
    if (!D) return -1.0;

    double cost_v = 1.0;
    double cost_h = 1.0;
    D[0] = 1.0;
    for (size_t i=1; i<n_x; i++) D[i * n_y] = D[(i-1) * n_y] + cost_v;
    for (size_t j=1; j<n_y; j++) D[j] = D[j-1] + cost_h;
    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            double cost_d = (_abs(x[i] - y[j]) <= eps ? 0.0 : 1.0);
            D[i * n_y + j] = _min3(D[(i-1) * n_y + j] + cost_v, D[i * n_y + (j-1)] + cost_h, D[(i-1) * n_y + (j-1)] + cost_d);
        }
    }
    result.distance = D[(n_x-1) * n_y + (n_y-1)];
    result.paths = D;

    return result;
}

DTW swale(const double* x, const double* y, size_t n_x, size_t n_y, double eps, double p, double r) {
    DTW result = {-1.0, NULL};
    double* D = malloc(n_x * n_y * sizeof(double));
    if (!D) return -1.0;

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
    if (!D) return -1.0;

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
    if (!D) return -1.0;

    D[0] = _abs(x[0] - y[0]);
    for (size_t i=1; i<n_x; i++) {
        double cost_v = (((x[i-1] <= x[i]) && (x[i] <= y[j])) || ((x[i-1] >= x[i]) && (x[i] >= y[j])) ? c : c + _min(_abs(x[i] - x[i-1], x[i] - y[j])));
        D[i * n_y] = D[(i-1) * n_y] + cost_v;
    }
    for (size_t j=1; j<n_y; j++) {
        double cost_h = (((y[j-1] <= y[j]) && (y[j] <= x[i])) || ((y[j-1] >= y[j]) && (y[j] >= x[i])) ? c : c + _min(_abs(y[j] - y[j-1], y[j] - x[i])));
        D[j] = D[j-1] + cost_h;
    }
    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            double cost_v = (((x[i-1] <= x[i]) && (x[i] <= y[j])) || ((x[i-1] >= x[i]) && (x[i] >= y[j])) ? c : c + _min(_abs(x[i] - x[i-1], x[i] - y[j])));
            double cost_h = (((y[j-1] <= y[j]) && (y[j] <= x[i])) || ((y[j-1] >= y[j]) && (y[j] >= x[i])) ? c : c + _min(_abs(y[j] - y[j-1], y[j] - x[i])));
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
    if (!D) return -1.0;

    D[0] = 0.0;
    for (size_t i=1; i<n_x; i++) D[i * n_y] = D[(i-1) * n_y] + _square(x[i] - x[i-1]) + nu + lambda;
    for (size_t j=1; j<n_y; j++) D[j] = D[j-1] + _square(y[i] - y[i-1]) + nu + lambda;
    for (size_t i=1; i<n_x; i++) {
        for (size_t j=1; j<n_y; j++) {
            double cost_v = _square(x[i] - x[i-1]) + nu + lambda;
            double cost_h = _square(y[i] - y[i-1]) + nu + lambda;
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
    Point* path = malloc((n_x + n_y) * sizeof(PathPoint));
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