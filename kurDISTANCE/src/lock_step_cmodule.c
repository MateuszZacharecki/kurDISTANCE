#define PY_SSIZE_T_CLEAN
#define ln_2 0.6931471805599453
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

// Category: Minkowski

double euclidean(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _square(x[i] - y[i]);
    return _sqrt(dist);
}

double manhattan(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _abs(x[i] - y[i]);
    return dist;
}

double minkowski(const double* x, const double* y, size_t n, size_t p) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _pow(_abs(x[i] - y[i]), p);
    return _sqrt_p(dist, p);
}

double chebyshev(const double* x, const double* y, size_t n) {
    double max = 0.0;
    for (size_t i=0; i<n; i++) {
        double abs = _abs(x[i] - y[i]);
        if (abs > max) max = abs;
    }
    return max;
}


// Category: L1

double sorensen(const double* x, const double* y, size_t n) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i=0; i<n; i++) {
        numerator += _abs(x[i] - y[i]);
        denominator += (x[i] + y[i]);
    }
    return numerator / denominator;
}

double gower(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _abs(x[i] - y[i]);
    return dist / n;
}

double soergel(const double* x, const double* y, size_t n) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i=0; i<n; i++) {
        numerator += _abs(x[i] - y[i]);
        denominator += _max(x[i], y[i]);
    }
    return numerator / denominator;
}

double kulczynski(const double* x, const double* y, size_t n) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i=0; i<n; i++) {
        numerator += _abs(x[i] - y[i]);
        denominator += _min(x[i], y[i]);
    }
    return numerator / denominator;
}

double canberra(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_abs(x[i] - y[i]) / (x[i] + y[i]));
    return dist;
}

double lorentzian(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_log(1 + _abs(x[i] - y[i])));
    return dist;
}


// Category: Intersection

double intersection(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _abs(x[i] - y[i]);
    return dist / 2;
}

double wave_hedges(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_abs(x[i] - y[i]) / _max(x[i], y[i]));
    return dist / 2;
}

double czekanowski(const double* x, const double* y, size_t n) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i=0; i<n; i++) {
        numerator += _min(x[i], y[i]);
        denominator += (x[i] + y[i]);
    }
    return 1 - 2 * numerator / denominator;
}

double motyka(const double* x, const double* y, size_t n) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i=0; i<n; i++) {
        numerator += _min(x[i], y[i]);
        denominator += (x[i] + y[i]);
    }
    return 1 - numerator / denominator;
}

double tanimoto(const double* x, const double* y, size_t n) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i=0; i<n; i++) {
        numerator += (_max(x[i], y[i]) - _min(x[i], y[i]));
        denominator += _max(x[i], y[i]);
    }
    return numerator / denominator;
}


// Category: Inner Product

double inner_product(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += x[i] * y[i];
    return dist;
}

double harmonic_mean(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (x[i] * y[i] / (x[i] + y[i]));
    return 2 * dist;
}

double kumar_hassebrook(const double* x, const double* y, size_t n) {
    double el1 = 0.0;
    double el2 = 0.0;
    for (size_t i=0; i<n; i++) {
        el1 += (x[i] * y[i]);
        el2 += (_square(x[i]) + _square(y[i]));
    }
    return el1 / (el2 - el1);
}

double jaccard(const double* x, const double* y, size_t n) {
    double el1 = 0.0;
    double el2 = 0.0;
    double el3 = 0.0;
    for (size_t i=0; i<n; i++) {
        el1 += (x[i] * y[i]);
        el2 += (_square(x[i]) + _square(y[i]));
        el3 += _square(x[i] - y[i]);
    }
    return el3 / (el2 - el1);
}

double cosine(const double* x, const double* y, size_t n) {
    double el1 = 0.0;
    double el2 = 0.0;
    double el3 = 0.0;
    for (size_t i=0; i<n; i++) {
        el1 += (x[i] * y[i]);
        el2 += _square(x[i]);
        el3 += _square(y[i]);
    }
    return 1 - el1 / (_sqrt(el2) * _sqrt(el3));
}

double dice(const double* x, const double* y, size_t n) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i=0; i<n; i++) {
        numerator += _square(x[i] - y[i]);
        denominator += (_square(x[i]) + _square(y[i]));
    }
    return 1 - numerator / denominator;
}


// Category: Square Chord

double fidelity(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _sqrt(x[i] * y[i]);
    return dist;
}

double bhattacharyya(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _sqrt(x[i] * y[i]);
    return _log(dist);
}

double squared_chord(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _square(_sqrt(x[i]) - _sqrt(y[i]));
    return dist;
}

double hellinger(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _square(_sqrt(x[i]) - _sqrt(y[i]));
    return _sqrt(2 * dist);
}

double matusita(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _square(_sqrt(x[i]) - _sqrt(y[i]));
    return _sqrt(dist);
}


// Category: Squared L2

double squared_euclidean(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _square(x[i] - y[i]);
    return dist;
}

double clark(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += _square(_abs(x[i] - y[i]) / (x[i] + y[i]));
    return _sqrt(dist);
}

double neyman_chisq(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_square(x[i] - y[i]) / x[i]);
    return dist;
}

double pearson_chisq(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_square(x[i] - y[i]) / y[i]);
    return dist;
}

double squared_chisq(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_square(x[i] - y[i]) / (x[i] + y[i]));
    return dist;
}

double divergence(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_square(x[i] - y[i]) / _square(x[i] + y[i]));
    return 2 * dist;
}

double additive_symmetric_chisq(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_square(x[i] - y[i]) * (x[i] + y[i]) / (x[i] * y[i]));
    return dist;
}

double probabilistic_symmetric_chisq(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_square(x[i] - y[i]) / (x[i] + y[i]));
    return 2 * dist;
}


// Category: Entropy

double kullback_leibler(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (x[i] * _log(x[i] / y[i]));
    return dist;
}

double jeffreys(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += ((x[i] - y[i]) * _log(x[i] / y[i]));
    return dist;
}

double k_divergence(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (x[i] * _log(2 * x[i] / (x[i] + y[i])));
    return dist;
}

double topsoe(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (x[i] * _log(2 * x[i] / (x[i] + y[i])) + y[i] * _log(2 * y[i] / (x[i] + y[i])));
    return dist;
}

double jensen_shannon(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (x[i] * _log(2 * x[i] / (x[i] + y[i])) + y[i] * _log(2 * y[i] / (x[i] + y[i])));
    return dist / 2;
}

double jensen_difference(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += ((x[i] * _log(x[i]) + y[i] * _log(y[i])) / 2 - (x[i] + y[i]) / 2 * _log((x[i] + y[i]) / 2));
    return dist;
}


// Category: Vicissitude

double vicis_wave_hedges(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_abs(x[i] - y[i]) / _min(x[i], y[i]));
    return dist;
}

double emanon_2(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_square(x[i] - y[i]) / _square(_min(x[i], y[i])));
    return dist;
}

double emanon_3(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_square(x[i] - y[i]) / _min(x[i], y[i]));
    return dist;
}

double emanon_4(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_square(x[i] - y[i]) / _max(x[i], y[i]));
    return dist;
}

double max_symmetric_chisq(const double* x, const double* y, size_t n) {
    double el1 = 0.0;
    double el2 = 0.0;
    for (size_t i=0; i<n; i++) {
        el1 += (_square(x[i] - y[i]) / x[i]);
        el2 += (_square(x[i] - y[i]) / y[i]);
    }
    return _max(el1, el2);
}

double min_symmetric_chisq(const double* x, const double* y, size_t n) {
    double el1 = 0.0;
    double el2 = 0.0;
    for (size_t i=0; i<n; i++) {
        el1 += (_square(x[i] - y[i]) / x[i]);
        el2 += (_square(x[i] - y[i]) / y[i]);
    }
    return _min(el1, el2);
}


// Category: Combination

double taneja(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += ((x[i] + y[i]) * _log((x[i] + y[i]) / (2 * _sqrt(x[i] * y[i]))));
    return dist / 2;
}

double kumar_johnson(const double* x, const double* y, size_t n) {
    double dist = 0.0;
    for (size_t i=0; i<n; i++) dist += (_square(_square(x[i]) - _square(y[i])) / _sqrt(_pow((x[i] * y[i]), 3)));
    return dist / 2;
}

double avg_l1_linf(const double* x, const double* y, size_t n) {
    double max = 0.0;
    double sum = 0.0;
    for (size_t i=0; i<n; i++) {
        double abs = _abs(x[i] - y[i]);
        if (abs > max) max = abs;
        sum += abs;
    }
    return (sum + max) / 2;
}


// ============================================================================
// PYTHON WRAPPER UTILITIES
// ============================================================================

static int extract_arrays(PyObject* args, const double** out_x, const double** out_y, size_t* out_n) {
    if (PyTuple_Size(args) < 2) {
        PyErr_Format(PyExc_RuntimeError, "Expected at least 2 arguments");
        return 0;
    }
    PyObject* arg0 = PyTuple_GetItem(args, 0);
    PyObject* arg1 = PyTuple_GetItem(args, 1);

    if (!PyArray_Check(arg0) || !PyArray_Check(arg1)) {
        PyErr_Format(PyExc_RuntimeError, "Expected numpy arrays");
        return 0;
    }

    PyArrayObject* arr1 = (PyArrayObject*)arg0;
    PyArrayObject* arr2 = (PyArrayObject*)arg1;

    if (PyArray_TYPE(arr1) != NPY_DOUBLE || PyArray_TYPE(arr2) != NPY_DOUBLE) {
        PyErr_Format(PyExc_RuntimeError, "Expected numpy double-typed arrays");
        return 0;
    }
    if (!PyArray_IS_C_CONTIGUOUS(arr1) || !PyArray_IS_C_CONTIGUOUS(arr2)) {
        PyErr_Format(PyExc_RuntimeError, "Expected contiguous arrays");
        return 0;
    }
    if (PyArray_SIZE(arr1) != PyArray_SIZE(arr2)) {
        PyErr_Format(PyExc_ValueError, "Expected same size arrays");
        return 0;
    }

    *out_x = PyArray_DATA(arr1);
    *out_y = PyArray_DATA(arr2);
    *out_n = PyArray_SIZE(arr1);
    return 1;
}

#define PY_DISTANCE_MEASURE(NAME) \
    static PyObject* py_##NAME(PyObject* self, PyObject* args) { \
        if (PyTuple_Size(args) != 2) \
            return PyErr_Format(PyExc_RuntimeError, "Expected 2 arguments (array, array)"); \
        const double *x, *y; \
        size_t n; \
        if (!extract_arrays(args, &x, &y, &n)) return NULL; \
        return PyFloat_FromDouble(NAME(x, y, n)); \
    }

PY_DISTANCE_MEASURE(euclidean)
PY_DISTANCE_MEASURE(manhattan)
PY_DISTANCE_MEASURE(chebyshev)
PY_DISTANCE_MEASURE(sorensen)
PY_DISTANCE_MEASURE(gower)
PY_DISTANCE_MEASURE(soergel)
PY_DISTANCE_MEASURE(kulczynski)
PY_DISTANCE_MEASURE(canberra)
PY_DISTANCE_MEASURE(lorentzian)
PY_DISTANCE_MEASURE(intersection)
PY_DISTANCE_MEASURE(wave_hedges)
PY_DISTANCE_MEASURE(czekanowski)
PY_DISTANCE_MEASURE(motyka)
PY_DISTANCE_MEASURE(tanimoto)
PY_DISTANCE_MEASURE(inner_product)
PY_DISTANCE_MEASURE(harmonic_mean)
PY_DISTANCE_MEASURE(kumar_hassebrook)
PY_DISTANCE_MEASURE(jaccard)
PY_DISTANCE_MEASURE(cosine)
PY_DISTANCE_MEASURE(dice)
PY_DISTANCE_MEASURE(fidelity)
PY_DISTANCE_MEASURE(bhattacharyya)
PY_DISTANCE_MEASURE(squared_chord)
PY_DISTANCE_MEASURE(hellinger)
PY_DISTANCE_MEASURE(matusita)
PY_DISTANCE_MEASURE(squared_euclidean)
PY_DISTANCE_MEASURE(clark)
PY_DISTANCE_MEASURE(neyman_chisq)
PY_DISTANCE_MEASURE(pearson_chisq)
PY_DISTANCE_MEASURE(squared_chisq)
PY_DISTANCE_MEASURE(divergence)
PY_DISTANCE_MEASURE(additive_symmetric_chisq)
PY_DISTANCE_MEASURE(probabilistic_symmetric_chisq)
PY_DISTANCE_MEASURE(kullback_leibler)
PY_DISTANCE_MEASURE(jeffreys)
PY_DISTANCE_MEASURE(k_divergence)
PY_DISTANCE_MEASURE(topsoe)
PY_DISTANCE_MEASURE(jensen_shannon)
PY_DISTANCE_MEASURE(jensen_difference)
PY_DISTANCE_MEASURE(vicis_wave_hedges)
PY_DISTANCE_MEASURE(emanon_2)
PY_DISTANCE_MEASURE(emanon_3)
PY_DISTANCE_MEASURE(emanon_4)
PY_DISTANCE_MEASURE(max_symmetric_chisq)
PY_DISTANCE_MEASURE(min_symmetric_chisq)
PY_DISTANCE_MEASURE(taneja)
PY_DISTANCE_MEASURE(kumar_johnson)
PY_DISTANCE_MEASURE(avg_l1_linf)

static PyObject* py_minkowski(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 3)
        return PyErr_Format(PyExc_RuntimeError, "Expected 3 arguments (array, array, p)");
    
    const double *x, *y;
    size_t n;
    
    if (!extract_arrays(args, &x, &y, &n)) return NULL;
    
    PyObject* arg2 = PyTuple_GetItem(args, 2);
    size_t p = PyLong_AsSize_t(arg2);
    if (PyErr_Occurred()) return NULL;
    
    return PyFloat_FromDouble(minkowski(x, y, n, p));
}


// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

static PyMethodDef lock_step_cmodule_methods[] = {
    {"euclidean", py_euclidean, METH_VARARGS, "Calculate Euclidean distance"},
    {"manhattan", py_manhattan, METH_VARARGS, "Calculate Manhattan distance"},
    {"minkowski", py_minkowski, METH_VARARGS, "Calculate Minkowski distance"},
    {"chebyshev", py_chebyshev, METH_VARARGS, "Calculate Chebyshev distance"},
    {"sorensen", py_sorensen, METH_VARARGS, "Calculate Sorensen distance"},
    {"gower", py_gower, METH_VARARGS, "Calculate Gower distance"},
    {"soergel", py_soergel, METH_VARARGS, "Calculate Soergel distance"},
    {"kulczynski", py_kulczynski, METH_VARARGS, "Calculate Kulczynski distance"},
    {"canberra", py_canberra, METH_VARARGS, "Calculate Canberra distance"},
    {"lorentzian", py_lorentzian, METH_VARARGS, "Calculate Lorentzian distance"},
    {"intersection", py_intersection, METH_VARARGS, "Calculate Intersection distance"},
    {"wave_hedges", py_wave_hedges, METH_VARARGS, "Calculate Wave Hedges distance"},
    {"czekanowski", py_czekanowski, METH_VARARGS, "Calculate Czekanowski distance"},
    {"motyka", py_motyka, METH_VARARGS, "Calculate Motyka distance"},
    {"tanimoto", py_tanimoto, METH_VARARGS, "Calculate Tanimoto distance"},
    {"inner_product", py_inner_product, METH_VARARGS, "Calculate Inner Product"},
    {"harmonic_mean", py_harmonic_mean, METH_VARARGS, "Calculate Harmonic Mean"},
    {"kumar_hassebrook", py_kumar_hassebrook, METH_VARARGS, "Calculate Kumar-Hassebrook distance"},
    {"jaccard", py_jaccard, METH_VARARGS, "Calculate Jaccard distance"},
    {"cosine", py_cosine, METH_VARARGS, "Calculate Cosine distance"},
    {"dice", py_dice, METH_VARARGS, "Calculate Dice distance"},
    {"fidelity", py_fidelity, METH_VARARGS, "Calculate Fidelity distance"},
    {"bhattacharyya", py_bhattacharyya, METH_VARARGS, "Calculate Bhattacharyya distance"},
    {"squared_chord", py_squared_chord, METH_VARARGS, "Calculate Squared-chord distance"},
    {"hellinger", py_hellinger, METH_VARARGS, "Calculate Hellinger distance"},
    {"matusita", py_matusita, METH_VARARGS, "Calculate Matusita distance"},
    {"squared_euclidean", py_squared_euclidean, METH_VARARGS, "Calculate Squared Euclidean distance"},
    {"clark", py_clark, METH_VARARGS, "Calculate Clark distance"},
    {"neyman_chisq", py_neyman_chisq, METH_VARARGS, "Calculate Neyman Chi-Square distance"},
    {"pearson_chisq", py_pearson_chisq, METH_VARARGS, "Calculate Pearson Chi-Square distance"},
    {"squared_chisq", py_squared_chisq, METH_VARARGS, "Calculate Squared Chi-Square distance"},
    {"divergence", py_divergence, METH_VARARGS, "Calculate Divergence distance"},
    {"additive_symmetric_chisq", py_additive_symmetric_chisq, METH_VARARGS, "Calculate Additive Symmmetric Chi-Square distance"},
    {"probabilistic_symmetric_chisq", py_probabilistic_symmetric_chisq, METH_VARARGS, "Calculate Probabilistic Symmetric Chi-Square distance"},
    {"kullback_leibler", py_kullback_leibler, METH_VARARGS, "Calculate Kullback-Leibler distance"},
    {"jeffreys", py_jeffreys, METH_VARARGS, "Calculate Jeffreys distance"},
    {"k_divergence", py_k_divergence, METH_VARARGS, "Calculate K Divergence"},
    {"topsoe", py_topsoe, METH_VARARGS, "Calculate Topsoe distance"},
    {"jensen_shannon", py_jensen_shannon, METH_VARARGS, "Calculate Jensen Shannon distance"},
    {"jensen_difference", py_jensen_difference, METH_VARARGS, "Calculate Jensen Difference distance"},
    {"vicis_wave_hedges", py_vicis_wave_hedges, METH_VARARGS, "Calculate Vicis-Wave Hedges distance"},
    {"emanon_2", py_emanon_2, METH_VARARGS, "Calculate Emanon 2 distance"},
    {"emanon_3", py_emanon_3, METH_VARARGS, "Calculate Emanon 3 distance"},
    {"emanon_4", py_emanon_4, METH_VARARGS, "Calculate Emanon 4 distance"},
    {"max_symmetric_chisq", py_max_symmetric_chisq, METH_VARARGS, "Calculate Max Symmetric Chi-Square distance"},
    {"min_symmetric_chisq", py_min_symmetric_chisq, METH_VARARGS, "Calculate Min-Symmetric Chi-Square distance"},
    {"taneja", py_taneja, METH_VARARGS, "Calculate Taneja distance"},
    {"kumar_johnson", py_kumar_johnson, METH_VARARGS, "Calculate Kumar-Johnson distance"},
    {"avg_l1_linf", py_avg_l1_linf, METH_VARARGS, "Calculate Avg (L1, Linf) distance"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef lock_step_cmodule_module = {
    PyModuleDef_HEAD_INIT,
    "lock_step_cmodule", 
    "Lock-Step Time-Series Distance Measures including 
    Minkowski-based measures, L1 Functions, Intersection Functions, 
    Inner Product Functions, Squared Chord Functions, Squared L2 Functions, 
    Shannon's Entropy Functions, Vicissitude Functions, Combination Functions.",
    -1,
    lock_step_cmodule_methods
};

PyMODINIT_FUNC PyInit_lock_step_cmodule() {
    PyObject* mod = PyModule_Create(&lock_step_cmodule_module);
    if (!mod) return NULL;
    import_array(); 
    return mod;
}




