#include <cmath>
#include <iostream>
using namespace std;
void cufd(double *res, double *d_Cp, const double *Cp, const double *Cs,
          const double *Den, string dir, int calc_id);

void forward(double *res, const double *Cp, const double *Cs, const double *Den,
             string dir) {
  cufd(res, NULL, Cp, Cs, Den, dir, 0);
}

void backward(double *d_Cp, const double *Cp, const double *Cs,
              const double *Den, string dir) {
  cufd(NULL, d_Cp, Cp, Cs, Den, dir, 1);
}