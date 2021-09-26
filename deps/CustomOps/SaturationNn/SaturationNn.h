// Dongzhuo Li 04/20/2019
#ifndef SATOP_H__
#define SATOP_H__

#include <math.h>
#include <algorithm>
#include "adept.h"
#include "adept_arrays.h"
std::mutex mu;  

using namespace adept;

#define EPSILON 1e-16
#define MAX_LINE_SEARCH 10000
#define NEWTON_TOL 1e-6
// #define DEBUG
// #define OUTPUT_AMG
// #define ALPHA 0.006323996017182
#define ALPHA 1.0
// #define ALPHA 0.001127

#define s0(z, x) s0[(z) * (nx) + (x)]      // row major
#define sref(z, x) sref[(z) * (nx) + (x)]  // row major
#define pt(z, x) pt[(z) * (nx) + (x)]      // potential
#define permi(z, x) permi[(z) * (nx) + (x)]
#define poro(z, x) poro[(z) * (nx) + (x)]
#define dporodt(z, x) dporodt[(z) * (nx) + (x)]
#define qw(z, x) qw[(z) * (nx) + (x)]
#define qo(z, x) qo[(z) * (nx) + (x)]
#define ij2ind(z, x) ((z) * (nx) + (x))
#define grad_s0(z, x) grad_s0[(z) * (nx) + (x)]
#define grad_p(z, x) grad_p[(z) * (nx) + (x)]
#define grad_permi(z, x) grad_permi[(z) * (nx) + (x)]
#define grad_dporodt(z, x) grad_dporodt[(z) * (nx) + (x)]

#define sat(z, x) sat[(z) * (nx) + (x)]
#define grad_sat(z, x) grad_sat[(z) * (nx) + (x)]  // grad from input

#include <Eigen/Dense>
// #include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
// #include <Eigen/SparseLU>

// #include <unsupported/Eigen/SparseExtra> // For reading MatrixMarket files

// #include <amgcl/adapter/eigen.hpp>  // DL 04/17/2019 use builtin
#include <amgcl/amg.hpp>
// #include <amgcl/backend/builtin.hpp>  // DL 04/17/2019 use builtin
#include <amgcl/backend/eigen.hpp>  // DL commented
// #include <amgcl/coarsening/smoothed_aggregation.hpp>
// #include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
// AMGCL_USE_EIGEN_VECTORS_WITH_BUILTIN_BACKEND()  // DL 04/17/2019 use builtin


class NN{
  public:
    const int64 * config;
    const double * theta; 
    int m;
    int size;
    Stack* stack;
    std::vector<Array<2, double, true>*> W;
    std::vector<Array<1, double, true>*> b;

    std::vector<Array<2, double, true>*> y1;
    std::vector<Array<2, double, true>*> y2;
    Array<2, double, true> X;
    double *grad_theta;
    NN(Stack *stack, const int64 *config, const int n_config, const double*theta): 
        stack(stack),
        config(config), m(n_config), theta(theta)
    {
        X = Array<2, double, true>(1,1);
        if (config[0]!=1 || config[m-1]!=1){
          printf("ERROR: config[0] and config[end] must be 1\n");
          throw;
        }
        int k = 0;
        W.resize(m-1);
        b.resize(m-1);
        y1.resize(m-1);
        y2.resize(m-1);
        for(int i=0;i<m-1;i++){
          W[i] = new Array<2,double,true>(config[i], config[i+1]);
          b[i] = new Array<1, double,true>(config[i+1]);
          for(int p=0;p<config[i];p++)
              for(int q=0;q<config[i+1];q++)
                (*W[i])(p,q) = theta[k++];
          for(int p=0;p<config[i+1];p++) (*b[i])(p) = theta[k++];
        }
        size = k;
        grad_theta = new double[size];
    };

    ~NN(){
      for(int i=0;i<m-1;i++){
        delete W[i];
        delete b[i];
      }
      delete [] grad_theta;
    }

    adouble compute(){
      // for(int i=0;i<m-1;i++){
      //   y1[i] = new Array<2,double, true>(1, config[i+1]);
      //   y2[i] = new Array<2,double, true>(1, config[i+1]);
      //   if (i==0) 
      //       *(y1[i]) = X**(*W[i]);
      //   else 
      //       *(y1[i]) = (*y2[i-1])**(*W[i]);
      //   for(int p=0;p<1;p++){
      //     for(int q=0;q<config[i+1];q++)
      //       if (i==m-2)
      //         (*y2[i])(p,q) = ((*y1[i])(p,q) + (*b[i])(q));
      //       else
      //         (*y2[i])(p,q) = tanh((*y1[i])(p,q) + (*b[i])(q));
      //   }
      // }
      // return (*y2[m-2])(0,0);
      adouble x2 = X(0,0) * X(0,0);
      return x2.value();
    };

    double compMobi(double s){
        X(0,0) = s;
        adouble out = compute();
        double ret = out.value();
        for(int i=0;i<m-1;i++){
          delete y1[i];
          delete y2[i];
        }
        return ret;
    }

    double gradMobi(double s){
        X(0,0) = s;
        stack->new_recording();
        adouble out = compute();
        out.set_gradient(1.0);
        stack->compute_adjoint();
        double ret = X.get_gradient()(0,0);
        for(int i=0;i<m-1;i++){
          delete y1[i];
          delete y2[i];
        }
        return ret;
    }

    adouble compMobiAD(double s){
        X(0,0) = s;
        adouble out = compute();
        return out;
    }

    void extract_gradient(){
        int k = 0;
        for(int i=0;i<m-1;i++){
          auto grad_W = (*W[i]).get_gradient();
          for(int p=0;p<config[i];p++)
            for(int q=0;q<config[i+1];q++)
              grad_theta[k++] = grad_W(p,q);
          auto grad_b = (*b[i]).get_gradient();
          for(int p=0;p<config[i+1];p++)
            grad_theta[k++] = grad_b(p);
        }
    }
    
};


// viscosities are global variables;
double mu_w;
double mu_o;
double compMobiW(NN& nn,double s) { return nn.compMobi(s)/ mu_w; }
double gradMobiW(NN& nn,double s) { return nn.gradMobi(s) / mu_w; }
double compMobiO(NN& nn,double s) { return nn.compMobi(1-s) / mu_o; }
double gradMobiO(NN& nn,double s) { return nn.gradMobi(1-s) / mu_o; }


// double compMobiW(NN& nn,double s) { return s*s/ mu_w; }
// double gradMobiW(NN& nn,double s) { return 2*s / mu_w; }
// double compMobiO(NN& nn,double s) { return (1-s)*(1-s) / mu_o; }
// double gradMobiO(NN& nn,double s) { return 2*(1-s) / mu_o; }



// double compMobiW(NN& nn,double s) { return nn.compMobi(s); }
// double gradMobiW(NN& nn,double s) { return nn.gradMobi(s); }
// double compMobiO(NN& nn,double s) { return nn.compMobi(s); }
// double gradMobiO(NN& nn,double s) { return nn.gradMobi(s); }


adouble compMobiW_AD(NN& nn,double s) { return nn.compMobiAD(s)/mu_w; }
adouble compMobiO_AD(NN& nn,double s) { return nn.compMobiAD(1-s)/mu_o; }


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

/*
compute residual vector
*/
void compRes(NN&nnW, NN&nnO, Eigen::VectorXd &resEg, Eigen::MatrixXd &sEg, const double* dporodt, const double *s0,
             const double *pt, const double *permi, const double *poro,
             const double *qw, const double *qo, double dt, double h, int nz,
             int nx) {
  int idRow = 0;
  double permi_r = 0.0, permi_l = 0.0, permi_d = 0.0, permi_u = 0.0;
  double mobi_r = 0.0, mobi_l = 0.0, mobi_d = 0.0, mobi_u = 0.0;
  double F_r = 0.0, F_l = 0.0, F_d = 0.0, F_u = 0.0;
  double h2 = h * h;

  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      idRow = i * nx + j;  // row-major
      permi_r = 0.0;
      permi_l = 0.0;
      permi_d = 0.0;
      permi_u = 0.0;
      mobi_r = 0.0;
      mobi_l = 0.0;
      mobi_d = 0.0;
      mobi_u = 0.0;
      F_r = 0.0;
      F_l = 0.0;
      F_d = 0.0;
      F_u = 0.0;

      if (j + 1 <= nx - 1) {
        if (pt(i, j + 1) > pt(i, j)) {
          mobi_r = compMobiW(nnW,sEg(i, j + 1));
        } else {
          mobi_r = compMobiW(nnW,sEg(i, j));
        }
        permi_r = harmonicAve(permi(i, j), permi(i, j + 1));
        F_r = (pt(i, j + 1) - pt(i, j));
      }

      if (j - 1 >= 0) {
        if (pt(i, j - 1) > pt(i, j)) {
          mobi_l = compMobiW(nnW,sEg(i, j - 1));
        } else {
          mobi_l = compMobiW(nnW,sEg(i, j));
        }
        permi_l = harmonicAve(permi(i, j), permi(i, j - 1));
        F_l = (pt(i, j) - pt(i, j - 1));
      }

      if (i + 1 <= nz - 1) {
        if (pt(i + 1, j) > pt(i, j)) {
          mobi_d = compMobiW(nnW,sEg(i + 1, j));
        } else {
          mobi_d = compMobiW(nnW,sEg(i, j));
        }
        permi_d = harmonicAve(permi(i, j), permi(i + 1, j));
        F_d = (pt(i + 1, j) - pt(i, j));
      }

      if (i - 1 >= 0) {
        if (pt(i - 1, j) > pt(i, j)) {
          mobi_u = compMobiW(nnW,sEg(i - 1, j));
        } else {
          mobi_u = compMobiW(nnW,sEg(i, j));
        }
        permi_u = harmonicAve(permi(i, j), permi(i - 1, j));
        F_u = (pt(i, j) - pt(i - 1, j));
      }

      resEg(idRow) = poro(i, j) * (sEg(i, j) - s0(i, j)) + dt * sEg(i, j) * dporodt(i, j) -
                     dt * ALPHA / h2 *
                         (permi_d * mobi_d * F_d - permi_u * mobi_u * F_u +
                          permi_r * mobi_r * F_r - permi_l * mobi_l * F_l) -
                     dt * (qw(i, j) + qo(i, j) * compMobiW(nnW,sEg(i, j)) /
                                          (compMobiO(nnO,sEg(i, j)) + EPSILON));
    }
  }
}


void compRes_AD(NN&nnW, NN&nnO, Array<1, double, true> &resEg, Eigen::MatrixXd &sEg, const double* dporodt, const double *s0,
             const double *pt, const double *permi, const double *poro,
             const double *qw, const double *qo, double dt, double h, int nz,
             int nx) {
  int idRow = 0;
  double permi_r = 0.0, permi_l = 0.0, permi_d = 0.0, permi_u = 0.0;
  adouble mobi_r = 0.0, mobi_l = 0.0, mobi_d = 0.0, mobi_u = 0.0;
  double F_r = 0.0, F_l = 0.0, F_d = 0.0, F_u = 0.0;
  double h2 = h * h;

  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      idRow = i * nx + j;  // row-major
      permi_r = 0.0;
      permi_l = 0.0;
      permi_d = 0.0;
      permi_u = 0.0;
      mobi_r = 0.0;
      mobi_l = 0.0;
      mobi_d = 0.0;
      mobi_u = 0.0;
      F_r = 0.0;
      F_l = 0.0;
      F_d = 0.0;
      F_u = 0.0;

      if (j + 1 <= nx - 1) {
        if (pt(i, j + 1) > pt(i, j)) {
          mobi_r = compMobiW_AD(nnW,sEg(i, j + 1));
        } else {
          mobi_r = compMobiW_AD(nnW,sEg(i, j));
        }
        permi_r = harmonicAve(permi(i, j), permi(i, j + 1));
        F_r = (pt(i, j + 1) - pt(i, j));
      }

      if (j - 1 >= 0) {
        if (pt(i, j - 1) > pt(i, j)) {
          mobi_l = compMobiW_AD(nnW,sEg(i, j - 1));
        } else {
          mobi_l = compMobiW_AD(nnW,sEg(i, j));
        }
        permi_l = harmonicAve(permi(i, j), permi(i, j - 1));
        F_l = (pt(i, j) - pt(i, j - 1));
      }

      if (i + 1 <= nz - 1) {
        if (pt(i + 1, j) > pt(i, j)) {
          mobi_d = compMobiW_AD(nnW,sEg(i + 1, j));
        } else {
          mobi_d = compMobiW_AD(nnW,sEg(i, j));
        }
        permi_d = harmonicAve(permi(i, j), permi(i + 1, j));
        F_d = (pt(i + 1, j) - pt(i, j));
      }

      if (i - 1 >= 0) {
        if (pt(i - 1, j) > pt(i, j)) {
          mobi_u = compMobiW_AD(nnW,sEg(i - 1, j));
        } else {
          mobi_u = compMobiW_AD(nnW,sEg(i, j));
        }
        permi_u = harmonicAve(permi(i, j), permi(i - 1, j));
        F_u = (pt(i, j) - pt(i - 1, j));
      }

      resEg(idRow) = poro(i, j) * (sEg(i, j) - s0(i, j)) + dt * sEg(i, j) * dporodt(i, j) -
                     dt * ALPHA / h2 *
                         (permi_d * mobi_d * F_d - permi_u * mobi_u * F_u +
                          permi_r * mobi_r * F_r - permi_l * mobi_l * F_l) -
                     dt * (qw(i, j) + qo(i, j) * compMobiW_AD(nnW,sEg(i, j)) /
                                          (compMobiO_AD(nnO,sEg(i, j)) + EPSILON));
    }
  }
}

/*
assemble Jacobian matrix
*/
void assembleJ(NN&nnW, NN& nnO, Eigen::SparseMatrix<double, Eigen::RowMajor> &Jac,
               Eigen::MatrixXd &sEg, const double* dporodt, const double *pt, const double *permi,
               const double *poro, const double *qo, double dt, double h,
               int nz, int nx) {
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(5);

  int idRow = 0;
  double permi_r = 0.0, permi_l = 0.0, permi_d = 0.0, permi_u = 0.0;
  double mobi_r = 0.0, mobi_l = 0.0, mobi_d = 0.0, mobi_u = 0.0;
  double F_r = 0.0, F_l = 0.0, F_d = 0.0, F_u = 0.0;
  double h2 = h * h;
  double coef = 0.0;
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      idRow = i * nx + j;  // row-major
      permi_r = 0.0;
      permi_l = 0.0;
      permi_d = 0.0;
      permi_u = 0.0;
      mobi_r = 0.0;
      mobi_l = 0.0;
      mobi_d = 0.0;
      mobi_u = 0.0;
      F_r = 0.0;
      F_l = 0.0;
      F_d = 0.0;
      F_u = 0.0;
      coef = -ALPHA * dt / h2;

      if (j + 1 <= nx - 1) {
        permi_r = harmonicAve(permi(i, j), permi(i, j + 1));
        F_r = (pt(i, j + 1) - pt(i, j));
        if (pt(i, j + 1) > pt(i, j)) {
          tripletList.push_back(
              T(idRow, ij2ind(i, j + 1),
                coef * permi_r * F_r * gradMobiW(nnW,sEg(i, j + 1))));
        } else {
          tripletList.push_back(T(idRow, ij2ind(i, j),
                                  coef * permi_r * F_r * gradMobiW(nnW,sEg(i, j))));
        }
      }

      if (j - 1 >= 0) {
        permi_l = harmonicAve(permi(i, j), permi(i, j - 1));
        F_l = (pt(i, j) - pt(i, j - 1));
        if (pt(i, j - 1) > pt(i, j)) {
          tripletList.push_back(
              T(idRow, ij2ind(i, j - 1),
                -coef * permi_l * F_l * gradMobiW(nnW,sEg(i, j - 1))));
        } else {
          tripletList.push_back(
              T(idRow, ij2ind(i, j),
                -coef * permi_l * F_l * gradMobiW(nnW,sEg(i, j))));
        }
      }

      if (i + 1 <= nz - 1) {
        permi_d = harmonicAve(permi(i, j), permi(i + 1, j));
        F_d = (pt(i + 1, j) - pt(i, j));
        if (pt(i + 1, j) > pt(i, j)) {
          tripletList.push_back(
              T(idRow, ij2ind(i + 1, j),
                coef * permi_d * F_d * gradMobiW(nnW,sEg(i + 1, j))));
        } else {
          tripletList.push_back(T(idRow, ij2ind(i, j),
                                  coef * permi_d * F_d * gradMobiW(nnW,sEg(i, j))));
        }
      }

      if (i - 1 >= 0) {
        permi_u = harmonicAve(permi(i, j), permi(i - 1, j));
        F_u = (pt(i, j) - pt(i - 1, j));
        if (pt(i - 1, j) > pt(i, j)) {
          tripletList.push_back(
              T(idRow, ij2ind(i - 1, j),
                -coef * permi_u * F_u * gradMobiW(nnW,sEg(i - 1, j))));
        } else {
          tripletList.push_back(
              T(idRow, ij2ind(i, j),
                -coef * permi_u * F_u * gradMobiW(nnW,sEg(i, j))));
        }
      }

      // coefficient before s(n+1)
      tripletList.push_back(T(idRow, ij2ind(i, j), poro(i, j) + dt * dporodt(i, j)));

      // coefficient before qo
      tripletList.push_back(
          T(idRow, ij2ind(i, j),
            -dt * qo(i, j) *
                (compMobiO(nnO,sEg(i, j)) * gradMobiW(nnW,sEg(i, j)) -
                 compMobiW(nnW,sEg(i, j)) * gradMobiO(nnO,sEg(i, j))) /
                pow((compMobiO(nnO,sEg(i, j)) + EPSILON), 2)));
    }
  }
  Jac.setFromTriplets(tripletList.begin(), tripletList.end());
}

/*
Back-tracking line search
*/
double linesearch(NN&nnW, NN&nnO, Eigen::MatrixXd &sEg, const Eigen::MatrixXd &delta_sEg_mat,
                  Eigen::MatrixXd &sEg_update, const double *dporodt, const double *s0, 
                  const double *pt, const double *permi, const double *poro,
                  const double *qw, const double *qo, double dt, double h,
                  int nz, int nx) {
  double alpha = 1.0;
  Eigen::VectorXd resEg(nz * nx);

  compRes(nnW, nnO, resEg, sEg, dporodt, s0, pt, permi, poro, qw, qo, dt, h, nz, nx);
  double res_norm = resEg.lpNorm<Eigen::Infinity>();
  // double res_norm = resEg.norm();

  sEg_update = sEg + alpha * delta_sEg_mat;
  compRes(nnW, nnO, resEg, sEg_update, dporodt, s0,  pt, permi, poro, qw, qo, dt, h, nz, nx);
  double res_norm_update = resEg.lpNorm<Eigen::Infinity>();
  // double res_norm_update = resEg.norm();
  int count = 0;

  while (res_norm_update > res_norm || sEg_update.minCoeff() < 0.0 ||
         sEg_update.maxCoeff() > 1.0) {
    if (count > MAX_LINE_SEARCH) {
      std::cout << "Linesearch failure!!!!!" << std::endl;
      exit(1);
    }
    alpha = 0.5 * alpha;
    sEg_update = sEg + alpha * delta_sEg_mat;
    compRes(nnW, nnO, resEg, sEg_update, dporodt, s0,  pt, permi, poro, qw, qo, dt, h, nz, nx);
    res_norm_update = resEg.lpNorm<Eigen::Infinity>();
    // res_norm_update = resEg.norm();
#ifdef DEBUG
    std::cout << "count = " << count << std::endl;
#endif
    count++;
  }
  return res_norm_update;
}

/*
forward function
*/
void forward(
             const double * thetaw, const int64 * configw, int mw, 
             const double * thetao, const int64 * configo, int mo, 
             double *sat, const double* dporodt, const double *s0, const double *pt,
             const double *permi, const double *poro, const double *qw,
             const double *qo, const double *sref, double dt, double h, int nz,
             int nx) {
  Stack stack;
  NN nnW(&stack, configw, mw, thetaw);
  NN nnO(&stack, configo, mo, thetao);

  Eigen::VectorXd resEg(nz * nx);     // residual vector in eigen
  Eigen::VectorXd negResEg(nz * nx);  // negative of resEg
  Eigen::MatrixXd sEg(nz, nx);
  Eigen::MatrixXd sEg_old(nz, nx);
  Eigen::MatrixXd delta_sEg_mat_tran(nx, nz);
  Eigen::MatrixXd delta_sEg_mat(nz, nx);

  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      sEg(i, j) = sref(i, j);
    }
  }

  Eigen::VectorXd delta_sEg = Eigen::VectorXd::Zero(nz * nx);
  Eigen::SparseMatrix<double, Eigen::RowMajor> Jac(nz * nx, nz * nx);

  double res_norm = 2.0, res_norm_old = 1.0;

  // Setup the solver:
  typedef amgcl::make_solver<
      amgcl::amg<amgcl::backend::eigen<double>, amgcl::coarsening::ruge_stuben,
                 amgcl::relaxation::spai0>,
      amgcl::solver::bicgstab<amgcl::backend::eigen<double>>>
      Solver;
  // // DL 04/17/2019 builtin
  // typedef amgcl::make_solver<
  //     amgcl::amg<amgcl::backend::builtin<double>,
  //                amgcl::coarsening::ruge_stuben, amgcl::relaxation::spai0>,
  //     amgcl::solver::bicgstab<amgcl::backend::builtin<double> > >
  //     Solver;

  // start Newton iterations
  // prevent division by zero
  while (fabs(res_norm - res_norm_old) > NEWTON_TOL * res_norm_old) {
    // while (res_norm > NEWTON_TOL) {
    res_norm_old = res_norm;

    compRes(nnW, nnO, resEg, sEg, dporodt, s0, pt, permi, poro, qw, qo, dt, h, nz, nx);
    assembleJ(nnW, nnO, Jac, sEg, dporodt, pt, permi, poro, qo, dt, h, nz, nx);
    sEg_old = sEg;
// ================ AMG ====================
#ifdef DEBUG
    std::cout << "SatOp--" << __LINE__ << std::endl;
#endif
    Solver::params prm;
    prm.solver.tol = 1e-20;
    Solver solve(Jac);
#ifdef OUTPUT_AMG
    std::cout << solve << std::endl;
#endif

    // Solve the system for the given RHS:
    int iters;
    double error;
    negResEg = -1.0 * resEg;
    std::tie(iters, error) = solve(negResEg, delta_sEg);
    // std::cout << "error"
    //           << " " << (Jac * delta_sEg - resEg).norm() / resEg.norm()
    //           << std::endl;
    // std::cout << "***" << iters << " " << error << std::endl;
#ifdef OUTPUT_AMG
    std::cout << iters << " " << error << std::endl;
#endif
    // =============================================

    // ============ Eigen SparseLU =============
    // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    // // Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>  solver;
    // solver.analyzePattern(Jac);
    // solver.compute(Jac);
    // if (solver.info() != Eigen::Success) {
    //   // decomposition failed
    //   std::cout << "!!!decomposition failed" << std::endl;
    //   exit(1);
    // }
    // negResEg = -1.0 * resEg;
    // delta_sEg = solver.solve(negResEg);
    // if (solver.info() != Eigen::Success) {
    //   // solving failed
    //   std::cout << "!!!solving failed" << std::endl;
    //   exit(1);
    // }
    // ================= END =====================

    // sEg += delta_sEg;
    Eigen::Map<Eigen::MatrixXd> delta_sEg_mat_tran(delta_sEg.data(), nx, nz);
    delta_sEg_mat = delta_sEg_mat_tran.transpose();
    res_norm = linesearch(nnW, nnO, sEg_old, delta_sEg_mat, sEg, dporodt, s0,  pt, permi, poro, qw,
                          qo, dt, h, nz, nx);
#ifdef DEBUG
    printf("res_norm = %f\n", res_norm);
#endif
    // printf("res_norm = %g\n", res_norm);
  }
  initialArray(sat, nz * nx, 0.0);
  // std::cout << "Finish one step" << std::endl;
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      sat(i, j) = sEg(i, j);
    }
  }
}



/*
Backward computation of gradients
*/
void backward(
  double *grad_theta1, double *grad_theta2,
  const double * thetaw, const int64 * configw, int mw, 
  const double * thetao, const int64 * configo, int mo, 
  const double *grad_sat, const double *sat, const double *s0,
              const double* dporodt,
              const double *pt, const double *permi, const double *poro,
              const double *qw, const double *qo, double dt, double h, int nz,
              int nx, double *grad_s0, double *grad_dporodt, double *grad_pt, double *grad_permi,
              double *grad_poro) {
  Stack stack;
  
  NN nnW(&stack, configw, mw, thetaw);
  NN nnO(&stack, configo, mo, thetao);

  Eigen::MatrixXd sEg(nz, nx);
  Eigen::SparseMatrix<double, Eigen::RowMajor> Jac(nz * nx, nz * nx);
  Eigen::SparseMatrix<double, Eigen::RowMajor> Trans_Jac(nz * nx, nz * nx);
  Eigen::VectorXd rhs(nz * nx);
  Eigen::VectorXd negRhs(nz * nx);
  Eigen::VectorXd adjoint = Eigen::VectorXd::Zero(nz * nx);
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      sEg(i, j) = sat(i, j);
    }
  }
  for (int i = 0; i < nz * nx; i++) {
    rhs(i) = grad_sat[i];  // row-major
  }
  assembleJ(nnW, nnO, Jac, sEg, dporodt, pt, permi, poro, qo, dt, h, nz, nx);
  // ================ AMG ====================
  // Setup the solver:
  typedef amgcl::make_solver<
      amgcl::amg<amgcl::backend::eigen<double>, amgcl::coarsening::ruge_stuben,
                 amgcl::relaxation::spai0>,
      amgcl::solver::bicgstab<amgcl::backend::eigen<double>>>
      Solver;
  // // DL 04/17/2019 builtin
  // typedef amgcl::make_solver<
  //     amgcl::amg<amgcl::backend::builtin<double>,
  //                amgcl::coarsening::ruge_stuben, amgcl::relaxation::spai0>,
  //     amgcl::solver::bicgstab<amgcl::backend::builtin<double> > >
  //     Solver;
  Trans_Jac = Jac.transpose();
  Solver solve(Trans_Jac);
#ifdef OUTPUT_AMG
  std::cout << solve << std::endl;
#endif
  int iters;
  double error;
  negRhs = -1.0 * rhs;
  std::tie(iters, error) = solve(negRhs, adjoint);
  // std::cout << "error"
  //           << " " << (Trans_Jac * adjoint - rhs).norm() / rhs.norm()
  //           << std::endl;
#ifdef OUTPUT_AMG
  std::cout << iters << " " << error << std::endl;
#endif
  // ================= END =====================

  // ============ Eigen SparseLU =============
  // Trans_Jac = Jac.transpose();
  // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  // // Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>  solver;
  // solver.analyzePattern(Trans_Jac);
  // solver.compute(Trans_Jac);
  // if (solver.info() != Eigen::Success) {
  //   // decomposition failed
  //   std::cout << "!!!decomposition failed" << std::endl;
  //   exit(1);
  // }
  // negRhs = -1.0 * rhs;
  // adjoint = solver.solve(negRhs);
  // if (solver.info() != Eigen::Success) {
  //   // solving failed
  //   std::cout << "!!!solving failed" << std::endl;
  //   exit(1);
  // }
  // ================= END =====================

  // compute gradient with respect to s0
  initialArray(grad_s0, nz * nx, 0.0);
  initialArray(grad_poro, nz * nx, 0.0);
  initialArray(grad_dporodt, nz * nx, 0.0);
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      grad_s0[i * nx + j] = -poro(i, j) * adjoint(i * nx + j);
      grad_poro[i * nx + j] = (sat(i, j) - s0(i, j)) * adjoint(i * nx + j);
      grad_dporodt[i * nx + j] = dt * sat(i, j) * adjoint(i * nx + j);
    }
  }

#ifdef DEBUG
  std::cout << "SatOpBackward--" << __LINE__ << std::endl;
#endif
  // assemble R_pt and R_permi (gradient of R w.r.t pt and permi)
  Eigen::SparseMatrix<double, Eigen::RowMajor> R_pt(nz * nx, nz * nx);
  Eigen::SparseMatrix<double, Eigen::RowMajor> R_permi(nz * nx, nz * nx);
  typedef Eigen::Triplet<double> T;
  std::vector<T> tL_pt, tL_permi;
  tL_pt.reserve(5);
  tL_permi.reserve(5);

  int idRow = 0;
  double permi_r = 0.0, permi_l = 0.0, permi_d = 0.0, permi_u = 0.0;
  double mobi_r = 0.0, mobi_l = 0.0, mobi_d = 0.0, mobi_u = 0.0;
  double F_r = 0.0, F_l = 0.0, F_d = 0.0, F_u = 0.0;
  double h2 = h * h;
  double coef = 0.0;
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      idRow = i * nx + j;  // row-major
      permi_r = 0.0;
      permi_l = 0.0;
      permi_d = 0.0;
      permi_u = 0.0;
      mobi_r = 0.0;
      mobi_l = 0.0;
      mobi_d = 0.0;
      mobi_u = 0.0;
      F_r = 0.0;
      F_l = 0.0;
      F_d = 0.0;
      F_u = 0.0;
      coef = -ALPHA * dt / h2;

      if (j + 1 <= nx - 1) {
        permi_r = harmonicAve(permi(i, j), permi(i, j + 1));
        F_r = (pt(i, j + 1) - pt(i, j));
        if (pt(i, j + 1) > pt(i, j)) {
          mobi_r = compMobiW(nnW,sEg(i, j + 1));
        } else {
          mobi_r = compMobiW(nnW,sEg(i, j));
        }
        tL_pt.push_back(T(idRow, ij2ind(i, j + 1), coef * mobi_r * permi_r));
        tL_pt.push_back(T(idRow, ij2ind(i, j), -coef * mobi_r * permi_r));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j),
              coef * mobi_r * F_r *
                  grad_harAve(permi(i, j), permi(i, j + 1), true)));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j + 1),
              coef * mobi_r * F_r *
                  grad_harAve(permi(i, j), permi(i, j + 1), false)));
      }

      if (j - 1 >= 0) {
        permi_l = harmonicAve(permi(i, j), permi(i, j - 1));
        F_l = (pt(i, j) - pt(i, j - 1));
        if (pt(i, j - 1) > pt(i, j)) {
          mobi_l = compMobiW(nnW,sEg(i, j - 1));
        } else {
          mobi_l = compMobiW(nnW,sEg(i, j));
        }
        tL_pt.push_back(T(idRow, ij2ind(i, j - 1), coef * mobi_l * permi_l));
        tL_pt.push_back(T(idRow, ij2ind(i, j), -coef * mobi_l * permi_l));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j),
              -coef * mobi_l * F_l *
                  grad_harAve(permi(i, j), permi(i, j - 1), true)));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j - 1),
              -coef * mobi_l * F_l *
                  grad_harAve(permi(i, j), permi(i, j - 1), false)));
      }

      if (i + 1 <= nz - 1) {
        permi_d = harmonicAve(permi(i, j), permi(i + 1, j));
        F_d = (pt(i + 1, j) - pt(i, j));
        if (pt(i + 1, j) > pt(i, j)) {
          mobi_d = compMobiW(nnW,sEg(i + 1, j));
        } else {
          mobi_d = compMobiW(nnW,sEg(i, j));
        }
        tL_pt.push_back(T(idRow, ij2ind(i + 1, j), coef * mobi_d * permi_d));
        tL_pt.push_back(T(idRow, ij2ind(i, j), -coef * mobi_d * permi_d));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j),
              coef * mobi_d * F_d *
                  grad_harAve(permi(i, j), permi(i + 1, j), true)));
        tL_permi.push_back(
            T(idRow, ij2ind(i + 1, j),
              coef * mobi_d * F_d *
                  grad_harAve(permi(i, j), permi(i + 1, j), false)));
      }

      if (i - 1 >= 0) {
        permi_u = harmonicAve(permi(i, j), permi(i - 1, j));
        F_u = (pt(i, j) - pt(i - 1, j));
        if (pt(i - 1, j) > pt(i, j)) {
          mobi_u = compMobiW(nnW,sEg(i - 1, j));
        } else {
          mobi_u = compMobiW(nnW,sEg(i, j));
        }
        tL_pt.push_back(T(idRow, ij2ind(i - 1, j), coef * mobi_u * permi_u));
        tL_pt.push_back(T(idRow, ij2ind(i, j), -coef * mobi_u * permi_u));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j),
              -coef * mobi_u * F_u *
                  grad_harAve(permi(i, j), permi(i - 1, j), true)));
        tL_permi.push_back(
            T(idRow, ij2ind(i - 1, j),
              -coef * mobi_u * F_u *
                  grad_harAve(permi(i, j), permi(i - 1, j), false)));
      }
    }
  }

#ifdef DEBUG
  std::cout << "SatOpBackward--" << __LINE__ << std::endl;
#endif
  R_pt.setFromTriplets(tL_pt.begin(), tL_pt.end());
  R_permi.setFromTriplets(tL_permi.begin(), tL_permi.end());

#ifdef DEBUG
  std::cout << "SatOpBackward--" << __LINE__ << std::endl;
#endif
  Eigen::VectorXd Trans_J_pt(nz * nx);
  Eigen::VectorXd Trans_J_permi(nz * nx);
  Trans_J_pt = R_pt.transpose() * adjoint;
  Trans_J_permi = R_permi.transpose() * adjoint;
#ifdef DEBUG
  std::cout << "SatOpBackward--" << __LINE__ << std::endl;
#endif
  initialArray(grad_pt, nz * nx, 0.0);
  initialArray(grad_permi, nz * nx, 0.0);
  for (int i = 0; i < nz * nx; i++) {
    grad_pt[i] = Trans_J_pt(i);
    grad_permi[i] = Trans_J_permi(i);
  }

  Array<1, double, true> resEg(nz*nx);
  stack.new_recording();
  compRes_AD(nnW, nnO, resEg, sEg, dporodt, s0, pt, permi, poro, qw, qo, dt, h, nz, nx);
  adouble l = 0.0;
  for (int i=0;i<nz*nx;i++) l += resEg(i) * adjoint[i];
  l.set_gradient(1.0);
  stack.compute_adjoint();
  nnW.extract_gradient();
  nnO.extract_gradient();
  for(int i=0;i<nnW.size;i++) grad_theta1[i] = nnW.grad_theta[i];
  for(int j=0;j<nnO.size;j++) grad_theta2[j] = nnO.grad_theta[j];

#ifdef DEBUG
  std::cout << "SatOpBackward--" << __LINE__ << std::endl;
#endif
}


#endif