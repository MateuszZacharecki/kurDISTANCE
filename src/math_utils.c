#include "math_utils.h"

double _abs(double x) {
    return (x < 0.0) ? -x : x;
}

double _sqrt(double x) {
    double x_i = x;
    double y_i = 1.0;
    double eps = 0.000001;
    while (_abs(x_i - y_i) > eps) {
        x_i = (x_i + y_i) / 2;
        y_i = x / x_i;
    }
    return x_i;
}

double _square(double x) {
    return x * x;
}

double _pow(double x, size_t p) {
    double pow = 1.0;
    for (size_t i=0; i<p; i++) pow *= x;
    return pow;
}

double _sqrt_p(double x, size_t p) {
    double x_i = x / p;
    double delta = 1.0;
    double eps = 0.000001;
    while (_abs(delta) > eps) {
        double pow = _pow(x_i, p - 1);
        delta = (pow * x_i - x) / (p * pow);
        x_i = x_i - delta;
    }
    return x_i;
}

double _log(double x) {
    if (x <= 0.0) return 0.0;
    if (x == 1.0) return 0.0;
    double ln_m = 0.0;
    int k = 0;
    while (x > 2.0) {
        x /= 2.0;
        k++;
    }
    while (x < 1.0) {
        x *= 2.0;
        k--;
    }
    double numerator = (x - 1.0) / (x + 1.0);
    double alpha = _square(numerator);
    double denominator = 1.0;
    for (size_t i=0; i<12; i++) {
        ln_m += numerator / denominator;
        numerator *= alpha;
        denominator += 2.0;
    }
    ln_m = 2.0 * ln_m;
    return ln_m + (k * ln_2);
}

double _min(double x, double y) {
    return (x > y) ? y : x;
}

double _max(double x, double y) {
    return (x > y) ? x : y;
}

double _min3(double x, double y, double z) {
    return _min(_min(x, y), z);
}

int _sgn(double x) {
    if (x > 0) return 1;
    else if (x < 0) return -1;
    else return 0;
}