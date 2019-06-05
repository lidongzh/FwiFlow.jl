// Dongzhuo Li 04/08/2019
#ifndef UPW_LAP_H__
#define UPW_LAP_H__

#include <math.h>
#define EPSILON 1e-16

// double array row major
// double array uninitialized
#define perm(z, x) perm[(z) * (nx) + (x)]  // row major
#define mobi(z, x) mobi[(z) * (nx) + (x)]  // row major
#define func(z, x) func[(z) * (nx) + (x)]  // row major
#define ij2ind(z, x) ((z) * (nx) + (x))
#define out(z, x) out[(z) * (nx) + (x)]
#define grad_out(z, x) grad_out[(z) * (nx) + (x)]
#define grad_perm(z, x) grad_perm[(z) * (nx) + (x)]
#define grad_mobi(z, x) grad_mobi[(z) * (nx) + (x)]
#define grad_func(z, x) grad_func[(z) * (nx) + (x)]

void initialArray(double *ip, int size, double value) {
  for (int i = 0; i < size; i++) {
    ip[i] = value;
    // printf("value = %f\n", value);
  }
}

double harmonicAve(double a, double b) {
  return 2.0 * (a * b) / (a + b + EPSILON);
}

double grad_harAve(double a, double b, bool isLeft) {
  if (isLeft)
    return 2.0 * (b * (a + b + EPSILON) - a * b) / pow((a + b + EPSILON), 2.0);
  else
    return 2.0 * (a * (a + b + EPSILON) - a * b) / pow((a + b + EPSILON), 2.0);
}

double bndAve(double a, double b) { return a * b / (2 * b - a + EPSILON); }

double grad_bndAve(double a, double b, bool isLeft) {
  if (isLeft)
    return (2 * pow(b, 2.0) + EPSILON * b) / pow((2 * b - a + EPSILON), 2.0);
  else
    return (-pow(a, 2.0) + EPSILON * a) / pow((2 * b - a + EPSILON), 2.0);
}

void forward(double *out, const double *perm, const double *mobi,
             const double *func, double h, double rhograv, int nz, int nx) {
  double perm_r = 0.0, perm_l = 0.0, perm_d = 0.0, perm_u = 0.0;
  double mobi_r = 0.0, mobi_l = 0.0, mobi_d = 0.0, mobi_u = 0.0;
  double F_r = 0.0, F_l = 0.0, F_d = 0.0, F_u = 0.0;
  double h2 = h * h;
  initialArray(out, nz * nx, 0.0);
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      perm_r = 0.0;
      perm_l = 0.0;
      perm_d = 0.0;
      perm_u = 0.0;
      mobi_r = 0.0;
      mobi_l = 0.0;
      mobi_d = 0.0;
      mobi_u = 0.0;
      F_r = 0.0;
      F_l = 0.0;
      F_d = 0.0;
      F_u = 0.0;

      if (j + 1 <= nx - 1) {
        if (func(i, j + 1) > func(i, j)) {
          mobi_r = mobi(i, j + 1);
        } else {
          mobi_r = mobi(i, j);
        }
        perm_r = harmonicAve(perm(i, j), perm(i, j + 1)) / h2;
        F_r = (func(i, j + 1) - func(i, j));
      }

      if (j - 1 >= 0) {
        if (func(i, j - 1) > func(i, j)) {
          mobi_l = mobi(i, j - 1);
        } else {
          mobi_l = mobi(i, j);
        }
        perm_l = harmonicAve(perm(i, j), perm(i, j - 1)) / h2;
        F_l = (func(i, j) - func(i, j - 1));
      }

      if (i + 1 <= nz - 1) {
        if (func(i + 1, j) > func(i, j)) {
          mobi_d = mobi(i + 1, j);
        } else {
          mobi_d = mobi(i, j);
        }
        perm_d = harmonicAve(perm(i, j), perm(i + 1, j)) / h2;
        F_d = (func(i + 1, j) - func(i, j));
      } else {
        F_d = rhograv * h;
        // perm_d = harmonicAve(perm(i, j), bndAve(perm(i, j), perm(i - 1, j)))
        // / h2;
        perm_d = perm(i, j) / h2;
        mobi_d = mobi(i, j);
      }

      if (i - 1 >= 0) {
        if (func(i - 1, j) > func(i, j)) {
          mobi_u = mobi(i - 1, j);
        } else {
          mobi_u = mobi(i, j);
        }
        perm_u = harmonicAve(perm(i, j), perm(i - 1, j)) / h2;
        F_u = (func(i, j) - func(i - 1, j));
      } else {
        F_u = rhograv * h;
        // perm_u = harmonicAve(perm(i, j), bndAve(perm(i, j), perm(i + 1, j)))
        // / h2;
        perm_u = perm(i, j) / h2;
        mobi_u = mobi(i, j);
      }

      out(i, j) = perm_d * mobi_d * F_d - perm_u * mobi_u * F_u +
                  perm_r * mobi_r * F_r - perm_l * mobi_l * F_l;
    }
  }
}

void backward(const double *grad_out, const double *perm, const double *mobi,
              const double *func, double h, double rhograv, double *grad_perm,
              double *grad_mobi, double *grad_func, int nz, int nx) {
  initialArray(grad_perm, nz * nx, 0.0);
  initialArray(grad_mobi, nz * nx, 0.0);
  initialArray(grad_func, nz * nx, 0.0);

  double perm_r = 0.0, perm_l = 0.0, perm_d = 0.0, perm_u = 0.0;
  double mobi_r = 0.0, mobi_l = 0.0, mobi_d = 0.0, mobi_u = 0.0;
  double F_r = 0.0, F_l = 0.0, F_d = 0.0, F_u = 0.0;
  double h2 = h * h;

  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      perm_r = 0.0;
      perm_l = 0.0;
      perm_d = 0.0;
      perm_u = 0.0;
      mobi_r = 0.0;
      mobi_l = 0.0;
      mobi_d = 0.0;
      mobi_u = 0.0;
      F_r = 0.0;
      F_l = 0.0;
      F_d = 0.0;
      F_u = 0.0;

      if (j + 1 <= nx - 1) {
        perm_r = harmonicAve(perm(i, j), perm(i, j + 1)) / h2;
        F_r = (func(i, j + 1) - func(i, j));
        if (func(i, j + 1) > func(i, j)) {
          mobi_r = mobi(i, j + 1);
          grad_mobi(i, j + 1) += grad_out(i, j) * perm_r * F_r;
        } else {
          mobi_r = mobi(i, j);
          grad_mobi(i, j) += grad_out(i, j) * perm_r * F_r;
        }
        grad_perm(i, j) += grad_out(i, j) * mobi_r * F_r *
                           grad_harAve(perm(i, j), perm(i, j + 1), true) / h2;
        grad_perm(i, j + 1) += grad_out(i, j) * mobi_r * F_r *
                               grad_harAve(perm(i, j), perm(i, j + 1), false) /
                               h2;
        grad_func(i, j) += -grad_out(i, j) * perm_r * mobi_r;
        grad_func(i, j + 1) += grad_out(i, j) * perm_r * mobi_r;
      }

      if (j - 1 >= 0) {
        perm_l = harmonicAve(perm(i, j), perm(i, j - 1)) / h2;
        F_l = (func(i, j) - func(i, j - 1));
        if (func(i, j - 1) > func(i, j)) {
          mobi_l = mobi(i, j - 1);
          grad_mobi(i, j - 1) += -grad_out(i, j) * perm_l * F_l;
        } else {
          mobi_l = mobi(i, j);
          grad_mobi(i, j) += -grad_out(i, j) * perm_l * F_l;
        }
        grad_perm(i, j) += -grad_out(i, j) * mobi_l * F_l *
                           grad_harAve(perm(i, j), perm(i, j - 1), true) / h2;
        grad_perm(i, j - 1) += -grad_out(i, j) * mobi_l * F_l *
                               grad_harAve(perm(i, j), perm(i, j - 1), false) /
                               h2;
        grad_func(i, j) += -grad_out(i, j) * perm_l * mobi_l;
        grad_func(i, j - 1) += grad_out(i, j) * perm_l * mobi_l;
      }

      if (i + 1 <= nz - 1) {
        perm_d = harmonicAve(perm(i, j), perm(i + 1, j)) / h2;
        F_d = (func(i + 1, j) - func(i, j));
        if (func(i + 1, j) > func(i, j)) {
          mobi_d = mobi(i + 1, j);
          grad_mobi(i + 1, j) += grad_out(i, j) * perm_d * F_d;
        } else {
          mobi_d = mobi(i, j);
          grad_mobi(i, j) += grad_out(i, j) * perm_d * F_d;
        }
        grad_perm(i, j) += grad_out(i, j) * mobi_d * F_d *
                           grad_harAve(perm(i, j), perm(i + 1, j), true) / h2;
        grad_perm(i + 1, j) += grad_out(i, j) * mobi_d * F_d *
                               grad_harAve(perm(i, j), perm(i + 1, j), false) /
                               h2;
        grad_func(i, j) += -grad_out(i, j) * perm_d * mobi_d;
        grad_func(i + 1, j) += grad_out(i, j) * perm_d * mobi_d;
      } else {
        F_d = rhograv * h;
        // perm_d = harmonicAve(perm(i, j), bndAve(perm(i, j), perm(i - 1, j)))
        // / h2;
        perm_d = perm(i, j) / h2;
        mobi_d = mobi(i, j);
        grad_mobi(i, j) += grad_out(i, j) * perm_d * F_d;
        grad_perm(i, j) += grad_out(i, j) * mobi_d * F_d / h2;
      }

      if (i - 1 >= 0) {
        perm_u = harmonicAve(perm(i, j), perm(i - 1, j)) / h2;
        F_u = (func(i, j) - func(i - 1, j));
        if (func(i - 1, j) > func(i, j)) {
          mobi_u = mobi(i - 1, j);
          grad_mobi(i - 1, j) += -grad_out(i, j) * perm_u * F_u;
        } else {
          mobi_u = mobi(i, j);
          grad_mobi(i, j) += -grad_out(i, j) * perm_u * F_u;
        }
        grad_perm(i, j) += -grad_out(i, j) * mobi_u * F_u *
                           grad_harAve(perm(i, j), perm(i - 1, j), true) / h2;
        grad_perm(i - 1, j) += -grad_out(i, j) * mobi_u * F_u *
                               grad_harAve(perm(i, j), perm(i - 1, j), false) /
                               h2;
        grad_func(i, j) += -grad_out(i, j) * perm_u * mobi_u;
        grad_func(i - 1, j) += grad_out(i, j) * perm_u * mobi_u;

      } else {
        F_u = rhograv * h;
        // perm_u = harmonicAve(perm(i, j), bndAve(perm(i, j), perm(i + 1, j)))
        // / h2;
        perm_u = perm(i, j) / h2;
        mobi_u = mobi(i, j);
        grad_mobi(i, j) += -grad_out(i, j) * perm_u * F_u;
        grad_perm(i, j) += -grad_out(i, j) * mobi_u * F_u / h2;
      }
    }
  }
}

#endif
