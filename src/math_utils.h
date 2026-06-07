#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stddef.h>

#define ln_2 0.6931471805599453

double _abs(double x);
double _square(double x);
double _sqrt(double x);
double _pow(double x, size_t p);
double _sqrt_p(double x, size_t p);
double _log(double x);
double _min(double x, double y);
double _max(double x, double y);
double _min3(double x, double y, double z);
int _sgn(double x);

#endif