// Dongzhuo Li 04/14/2019
#ifndef UPWPSOP_H__
#define UPWPSOP_H__

#include <math.h>
#define EPSILON 1e-16

// double array row major
// double array uninitialized
#define permi(z, x) permi[(z) * (nx) + (x)]      // row major
#define mobi(z, x) mobi[(z) * (nx) + (x)]        // row major
#define src(z, x) src[(z) * (nx) + (x)]          // row major
#define funcref(z, x) funcref[(z) * (nx) + (x)]  // row major
#define ij2ind(z, x) ((z) * (nx) + (x))
#define grad_pres(z, x) grad_out[(z) * (nx) + (x)]
#define grad_permi(z, x) grad_permi[(z) * (nx) + (x)]
#define grad_mobi(z, x) grad_mobi[(z) * (nx) + (x)]
#define grad_src(z, x) grad_src[(z) * (nx) + (x)]
#define pres(z, x) pres[(z) * (nx) + (x)]

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

// #include <unsupported/Eigen/SparseExtra> // For reading MatrixMarket files

#include <amgcl/amg.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>

void intialArray(double *ip, int size, double value) {
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

void assembleMat(Eigen::SparseMatrix<double, Eigen::RowMajor> &Amat,
                 Eigen::VectorXd &rhs, const double *permi, const double *mobi,
                 const double *src, const double *funcref, double h,
                 double rhograv, int nz, int nx);

void forward(double *pres, const double *permi, const double *mobi,
             const double *src, const double *funcref, double h, double rhograv,
             int index, int nz, int nx) {
  Eigen::SparseMatrix<double, Eigen::RowMajor> Amat(nz * nx, nz * nx);
  Eigen::VectorXd rhs(nz * nx);
  Eigen::VectorXd pvec = Eigen::VectorXd::Zero(Amat.rows());

  // assemble matrix
  assembleMat(Amat, rhs, permi, mobi, src, funcref, h, rhograv, nz, nx);

  std::cout << "Forward: solving Poisson equation. Step: " << index
            << std::endl;

  if (index == 1) {
    Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>  solver;
    solver.analyzePattern(Amat);
    solver.compute(Amat);
    if (solver.info() != Eigen::Success) {
      // decomposition failed
      std::cout << "!!!decomposition failed" << std::endl;
      exit(1);
    }
    pvec = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
      // solving failed
      std::cout << "!!!solving failed" << std::endl;
      exit(1);
    }
    // std::cout << "#iterations: " << solver.iterations() << std::endl;
    // std::cout << "estimated error: " << solver.error() << std::endl;
  } else {
    // ================ AMG ====================
    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<amgcl::backend::eigen<double>,
                   amgcl::coarsening::smoothed_aggregation,
                   amgcl::relaxation::spai0>,
        amgcl::solver::bicgstab<amgcl::backend::eigen<double> > >
        Solver;

    Solver solve(Amat);
    std::cout << solve << std::endl;

    // Solve the system for the given RHS:
    int iters;
    double error;
    // Eigen::VectorXd x0 = Eigen::VectorXd::Zero(Amat.rows());
    std::tie(iters, error) = solve(rhs, pvec);

    std::cout << iters << " " << error << std::endl;
    // =============================================
  }

  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      pres[ij2ind(i, j)] = pvec(ij2ind(i, j));
    }
  }
}

void backward(const double *grad_pres, const double *pres, const double *permi,
              const double *mobi, const double *src, const double *funcref,
              double h, double rhograv, int index, int nz, int nx,
              double *grad_permi, double *grad_mobi, double *grad_src) {
  // F_p * p_v + F_v = 0
  // J_v = J_p * p_v = -J_p * F^{-1}_p * F_v = - s * F_v
  // F^T_p * s^T = J^T_p
  // J_v, J_p, s are row-vectors

  Eigen::SparseMatrix<double, Eigen::RowMajor> Amat(nz * nx, nz * nx);
  Eigen::SparseMatrix<double, Eigen::RowMajor> Trans_Amat(nz * nx, nz * nx);
  Eigen::VectorXd rhs(nz * nx);
  Eigen::VectorXd s = Eigen::VectorXd::Zero(Trans_Amat.rows());

  // assemble matrix
  assembleMat(Amat, rhs, permi, mobi, src, funcref, h, rhograv, nz, nx);
  for (int i = 0; i < nz * nx; i++) {
    rhs(i) = grad_pres[i];
  }
  Trans_Amat = Amat.transpose();

  if (index == 1) {
    Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>  solver;
    solver.analyzePattern(Trans_Amat);
    solver.compute(Trans_Amat);
    if (solver.info() != Eigen::Success) {
      // decomposition failed
      std::cout << "!!!decomposition failed" << std::endl;
      exit(1);
    }
    s = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
      // solving failed
      std::cout << "!!!solving failed" << std::endl;
      exit(1);
    }
  } else {
    // ================ AMG ====================
    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<amgcl::backend::eigen<double>,
                   amgcl::coarsening::smoothed_aggregation,
                   amgcl::relaxation::spai0>,
        amgcl::solver::bicgstab<amgcl::backend::eigen<double> > >
        Solver;

    Solver solve(Trans_Amat);
    std::cout << solve << std::endl;

    // Solve the system for the given RHS:
    int iters;
    double error;
    // Eigen::VectorXd x0 = Eigen::VectorXd::Zero(Amat.rows());
    std::tie(iters, error) = solve(rhs, s);

    std::cout << iters << " " << error << std::endl;
    // =============================================
  }

  s = s * h * h;

  // grad_g is actually s
  for (int i = 0; i < nz * nx; i++) {
    grad_src[i] = s(i);
  }

  // now compute grad_permi and grad_mobi
  Eigen::SparseMatrix<double, Eigen::RowMajor> F_permi(nz * nx, nz * nx);
  Eigen::SparseMatrix<double, Eigen::RowMajor> F_mobi(nz * nx, nz * nx);

  typedef Eigen::Triplet<double> T;
  std::vector<T> tL_permi, tL_mobi;
  tL_permi.reserve(5);
  tL_mobi.reserve(5);

  int idRow = 0;
  double h2 = h * h;
  double permi_r = 0.0, permi_l = 0.0, permi_d = 0.0, permi_u = 0.0;
  double mobi_r = 0.0, mobi_l = 0.0, mobi_d = 0.0, mobi_u = 0.0;
  double F_r = 0.0, F_l = 0.0, F_d = 0.0, F_u = 0.0;

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
        permi_r = harmonicAve(permi(i, j), permi(i, j + 1));
        F_r = pres(i, j) - pres(i, j + 1);
        // if (funcref(i, j + 1) > funcref(i, j)) {
        // 	mobi_r = mobi(i, j + 1);
        // 	tL_mobi.push_back(T(idRow, ij2ind(i, j + 1), permi_r * F_r));
        // } else {
        // 	mobi_r = mobi(i, j);
        // 	tL_mobi.push_back(T(idRow, ij2ind(i, j), permi_r * F_r));
        // }
        mobi_r = (mobi(i, j) + mobi(i, j + 1)) / 2.0;
        tL_mobi.push_back(T(idRow, ij2ind(i, j), 0.5 * permi_r * F_r));
        tL_mobi.push_back(T(idRow, ij2ind(i, j + 1), 0.5 * permi_r * F_r));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j),
              mobi_r * F_r * grad_harAve(permi(i, j), permi(i, j + 1), true)));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j + 1),
              mobi_r * F_r * grad_harAve(permi(i, j), permi(i, j + 1), false)));
      }

      if (j - 1 >= 0) {
        permi_l = harmonicAve(permi(i, j), permi(i, j - 1));
        F_l = pres(i, j) - pres(i, j - 1);
        // if (funcref(i, j - 1) > funcref(i, j)) {
        // 	mobi_l = mobi(i, j - 1);
        // 	tL_mobi.push_back(T(idRow, ij2ind(i, j - 1), permi_l * F_l));
        // } else {
        // 	mobi_l = mobi(i, j);
        // 	tL_mobi.push_back(T(idRow, ij2ind(i, j), permi_l * F_l));
        // }
        mobi_l = (mobi(i, j) + mobi(i, j - 1)) / 2.0;
        tL_mobi.push_back(T(idRow, ij2ind(i, j), 0.5 * permi_l * F_l));
        tL_mobi.push_back(T(idRow, ij2ind(i, j - 1), 0.5 * permi_l * F_l));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j),
              mobi_l * F_l * grad_harAve(permi(i, j), permi(i, j - 1), true)));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j - 1),
              mobi_l * F_l * grad_harAve(permi(i, j), permi(i, j - 1), false)));
      }

      if (i + 1 <= nz - 1) {
        permi_d = harmonicAve(permi(i, j), permi(i + 1, j));
        F_d = pres(i, j) - pres(i + 1, j);
        // if (funcref(i + 1, j) > funcref(i, j)) {
        // 	mobi_d = mobi(i + 1, j);
        // 	tL_mobi.push_back(T(idRow, ij2ind(i + 1, j), permi_d * F_d));
        // } else {
        // 	mobi_d = mobi(i, j);
        // 	tL_mobi.push_back(T(idRow, ij2ind(i, j), permi_d * F_d));
        // }
        mobi_d = (mobi(i, j) + mobi(i + 1, j)) / 2.0;
        tL_mobi.push_back(T(idRow, ij2ind(i, j), 0.5 * permi_d * F_d));
        tL_mobi.push_back(T(idRow, ij2ind(i + 1, j), 0.5 * permi_d * F_d));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j),
              mobi_d * F_d * grad_harAve(permi(i, j), permi(i + 1, j), true)));
        tL_permi.push_back(
            T(idRow, ij2ind(i + 1, j),
              mobi_d * F_d * grad_harAve(permi(i, j), permi(i + 1, j), false)));
      } else {
        // mobi_d = 1.5 * mobi(i, j) - 0.5 * mobi(i - 1, j);
        mobi_d = mobi(i, j);
        tL_permi.push_back(T(idRow, ij2ind(i, j), -h * rhograv * mobi_d));
        tL_mobi.push_back(T(idRow, ij2ind(i, j), -h * rhograv * permi(i, j)));
        // tL_mobi.push_back(
        //     T(idRow, ij2ind(i, j), -1.5 * h * rhograv * permi(i, j)));
        // tL_mobi.push_back(
        //     T(idRow, ij2ind(i - 1, j), 0.5 * h * rhograv * permi(i, j)));
      }

      if (i - 1 >= 0) {
        permi_u = harmonicAve(permi(i, j), permi(i - 1, j));
        F_u = pres(i, j) - pres(i - 1, j);
        // if (funcref(i - 1, j) > funcref(i, j)) {
        // 	mobi_u = mobi(i - 1, j);
        // 	tL_mobi.push_back(T(idRow, ij2ind(i - 1, j), permi_u * F_u));
        // } else {
        // 	mobi_u = mobi(i, j);
        // 	tL_mobi.push_back(T(idRow, ij2ind(i, j), permi_u * F_u));
        // }
        mobi_u = (mobi(i, j) + mobi(i - 1, j)) / 2.0;
        tL_mobi.push_back(T(idRow, ij2ind(i, j), 0.5 * permi_u * F_u));
        tL_mobi.push_back(T(idRow, ij2ind(i - 1, j), 0.5 * permi_u * F_u));
        tL_permi.push_back(
            T(idRow, ij2ind(i, j),
              mobi_u * F_u * grad_harAve(permi(i, j), permi(i - 1, j), true)));
        tL_permi.push_back(
            T(idRow, ij2ind(i - 1, j),
              mobi_u * F_u * grad_harAve(permi(i, j), permi(i - 1, j), false)));
      } else {
        // mobi_u = 1.5 * mobi(i, j) - 0.5 * mobi(i + 1, j);
        mobi_u = mobi(i, j);
        tL_permi.push_back(T(idRow, ij2ind(i, j), h * rhograv * mobi_u));
        tL_mobi.push_back(T(idRow, ij2ind(i, j), h * rhograv * permi(i, j)));
        // tL_mobi.push_back(
        //     T(idRow, ij2ind(i, j), 1.5 * h * rhograv * permi(i, j)));
        // tL_mobi.push_back(
        //     T(idRow, ij2ind(i + 1, j), -0.5 * h * rhograv * permi(i, j)));
      }
    }
  }

  F_permi.setFromTriplets(tL_permi.begin(), tL_permi.end());  // row-major
  F_mobi.setFromTriplets(tL_mobi.begin(), tL_mobi.end());     // row-major

  Eigen::VectorXd Trans_J_permi(nz * nx), Trans_J_mobi(nz * nx);
  // Eigen::SparseMatrix<double, Eigen::RowMajor> Trans_F_coef(nz * nx, nz *
  // nx); Trans_F_coef = F_coef.transpose();
  Trans_J_permi = -F_permi.transpose() * s;
  Trans_J_mobi = -F_mobi.transpose() * s;

  for (int i = 0; i < nz * nx; i++) {
    grad_permi[i] = Trans_J_permi(i);
    grad_mobi[i] = Trans_J_mobi(i);
  }
}

void assembleMat(Eigen::SparseMatrix<double, Eigen::RowMajor> &Amat,
                 Eigen::VectorXd &rhs, const double *permi, const double *mobi,
                 const double *src, const double *funcref, double h,
                 double rhograv, int nz, int nx) {
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(5);

  int idRow = 0;
  double T_r = 0.0, T_l = 0.0, T_d = 0.0, T_u = 0.0;
  double exT_d = 0.0, exT_u = 0.0;
  double permi_r = 0.0, permi_l = 0.0, permi_d = 0.0, permi_u = 0.0;
  double mobi_r = 0.0, mobi_l = 0.0, mobi_d = 0.0, mobi_u = 0.0;
  double h2 = h * h;
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      idRow = i * nx + j;  // row-major
      T_r = 0.0;
      T_l = 0.0;
      T_d = 0.0;
      T_u = 0.0;
      exT_d = 0.0;
      exT_u = 0.0;
      permi_r = 0.0;
      permi_l = 0.0;
      permi_d = 0.0;
      permi_u = 0.0;
      mobi_r = 0.0;
      mobi_l = 0.0;
      mobi_d = 0.0;
      mobi_u = 0.0;

      if (j + 1 <= nx - 1) {
        // if (funcref(i, j + 1) > funcref(i, j)) {
        // 	mobi_r = mobi(i, j + 1);
        // } else {
        // 	mobi_r = mobi(i, j);
        // }
        mobi_r = (mobi(i, j) + mobi(i, j + 1)) / 2.0;
        permi_r = harmonicAve(permi(i, j), permi(i, j + 1));
        T_r = permi_r * mobi_r;
        tripletList.push_back(T(idRow, ij2ind(i, j + 1), -T_r));
      }

      if (j - 1 >= 0) {
        // if (funcref(i, j - 1) > funcref(i, j)) {
        // 	mobi_l = mobi(i, j - 1);
        // } else {
        // 	mobi_l = mobi(i, j);
        // }
        mobi_l = (mobi(i, j) + mobi(i, j - 1)) / 2.0;
        permi_l = harmonicAve(permi(i, j), permi(i, j - 1));
        T_l = permi_l * mobi_l;
        tripletList.push_back(T(idRow, ij2ind(i, j - 1), -T_l));
      }

      if (i + 1 <= nz - 1) {
        // if (funcref(i + 1, j) > funcref(i, j)) {
        // 	mobi_d = mobi(i + 1, j);
        // } else {
        // 	mobi_d = mobi(i, j);
        // }
        mobi_d = (mobi(i, j) + mobi(i + 1, j)) / 2.0;
        permi_d = harmonicAve(permi(i, j), permi(i + 1, j));
        T_d = permi_d * mobi_d;
        tripletList.push_back(T(idRow, ij2ind(i + 1, j), -T_d));
      } else {
        exT_d = permi(i, j) * mobi(i, j);
        // exT_d = permi(i, j) * (1.5 * mobi(i, j) - 0.5 * mobi(i - 1, j));
      }

      if (i - 1 >= 0) {
        // if (funcref(i - 1, j) > funcref(i, j)) {
        // 	mobi_u = mobi(i - 1, j);
        // } else {
        // 	mobi_u = mobi(i, j);
        // }
        mobi_u = (mobi(i, j) + mobi(i - 1, j)) / 2.0;
        permi_u = harmonicAve(permi(i, j), permi(i - 1, j));
        T_u = permi_u * mobi_u;
        tripletList.push_back(T(idRow, ij2ind(i - 1, j), -T_u));
      } else {
        exT_u = permi(i, j) * mobi(i, j);
        // exT_u = permi(i, j) * (1.5 * mobi(i, j) - 0.5 * mobi(i + 1, j));
      }

      tripletList.push_back(T(idRow, ij2ind(i, j), T_r + T_l + T_d + T_u));

      rhs(idRow) = h2 * src(i, j) - exT_u * rhograv * h + exT_d * rhograv * h;
    }
  }
  tripletList.push_back(T(0, ij2ind(0, 0), 1.0));
  // tripletList.push_back(T(0, ij2ind(0, 0), coef(0, 0)));

  Amat.setFromTriplets(tripletList.begin(), tripletList.end());  // row-major
}

#endif
