// Dongzhuo Li 04/01/2019
#ifndef POISSONOP_H__
#define POISSONOP_H__

#define RHO_o 1000.0
#define GRAV 0.0
// double array row major
// double array uninitialized
#define coef(z, x) coef[(z) * (nx) + (x)]  // row major
#define g(z, x) g[(z) * (nx) + (x)]        // row major
#define ij2ind(z, x) ((z) * (nx) + (x))
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

void assembleMat(Eigen::SparseMatrix<double, Eigen::RowMajor> &Amat,
                 Eigen::VectorXd &rhs, const double *coef, const double *g,
                 double h, double rhograv, int nz, int nx);

void forward(double *pres, const double *coef, const double *g, double h,
             double rhograv, int index, int nz, int nx) {
  Eigen::SparseMatrix<double, Eigen::RowMajor> Amat(nz * nx, nz * nx);
  Eigen::VectorXd rhs(nz * nx);
  Eigen::VectorXd pvec = Eigen::VectorXd::Zero(Amat.rows());

  // assemble matrix
  assembleMat(Amat, rhs, coef, g, h, rhograv, nz, nx);

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

void backward(const double *grad_p, const double *pres, const double *coef,
              const double *g, double h, double rhograv, int index, int nz,
              int nx, double *grad_coef, double *grad_g) {
  // F_p * p_v + F_v = 0
  // J_v = J_p * p_v = -J_p * F^{-1}_p * F_v = - s * F_v
  // F^T_p * s^T = J^T_p
  // J_v, J_p, s are row-vectors

  Eigen::SparseMatrix<double, Eigen::RowMajor> Amat(nz * nx, nz * nx);
  Eigen::SparseMatrix<double, Eigen::RowMajor> Trans_Amat(nz * nx, nz * nx);
  Eigen::VectorXd rhs(nz * nx);
  Eigen::VectorXd s = Eigen::VectorXd::Zero(Trans_Amat.rows());
  double h2 = h * h;

  // assemble matrix
  assembleMat(Amat, rhs, coef, g, h, rhograv, nz, nx);
  for (int i = 0; i < nz * nx; i++) {
    rhs(i) = grad_p[i];
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

  // s = s * h * h;

  // grad_g is actually s
  for (int i = 0; i < nz * nx; i++) {
    grad_g[i] = s(i) * h2;
  }

  // now compute grad_coef
  Eigen::SparseMatrix<double, Eigen::RowMajor> F_coef(nz * nx, nz * nx);

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(5);

  int idRow = 0;
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      idRow = i * nx + j;  // row-major

      if (j + 1 <= nx - 1) {
        tripletList.push_back(
            T(idRow, ij2ind(i, j), 0.5 * (pres(i, j) - pres(i, j + 1))));
        tripletList.push_back(
            T(idRow, ij2ind(i, j + 1), 0.5 * (pres(i, j) - pres(i, j + 1))));
      }

      if (j - 1 >= 0) {
        tripletList.push_back(
            T(idRow, ij2ind(i, j), 0.5 * (pres(i, j) - pres(i, j - 1))));
        tripletList.push_back(
            T(idRow, ij2ind(i, j - 1), 0.5 * (pres(i, j) - pres(i, j - 1))));
      }

      if (i + 1 <= nz - 1) {
        tripletList.push_back(
            T(idRow, ij2ind(i, j), 0.5 * (pres(i, j) - pres(i + 1, j))));
        tripletList.push_back(
            T(idRow, ij2ind(i + 1, j), 0.5 * (pres(i, j) - pres(i + 1, j))));

      } else {
        tripletList.push_back(T(idRow, ij2ind(i, j), -1.5 * h * rhograv));
        tripletList.push_back(T(idRow, ij2ind(i - 1, j), 0.5 * h * rhograv));
      }

      if (i - 1 >= 0) {
        tripletList.push_back(
            T(idRow, ij2ind(i, j), 0.5 * (pres(i, j) - pres(i - 1, j))));
        tripletList.push_back(
            T(idRow, ij2ind(i - 1, j), 0.5 * (pres(i, j) - pres(i - 1, j))));

      } else {
        tripletList.push_back(T(idRow, ij2ind(i, j), 1.5 * h * rhograv));
        tripletList.push_back(T(idRow, ij2ind(i + 1, j), -0.5 * h * rhograv));
      }
    }
  }

  F_coef.setFromTriplets(tripletList.begin(), tripletList.end());  // row-major

  Eigen::VectorXd Trans_J_coef(nz * nx);
  Eigen::SparseMatrix<double, Eigen::RowMajor> Trans_F_coef(nz * nx, nz * nx);
  Trans_F_coef = F_coef.transpose();
  Trans_J_coef = -Trans_F_coef * s;

  std::cout << "Backward: solving Poisson equation. Step: " << index
            << std::endl;
  for (int i = 0; i < nz * nx; i++) {
    grad_coef[i] = Trans_J_coef(i);
  }
}

void assembleMat(Eigen::SparseMatrix<double, Eigen::RowMajor> &Amat,
                 Eigen::VectorXd &rhs, const double *coef, const double *g,
                 double h, double rhograv, int nz, int nx) {
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(5);

  int idRow = 0;
  double T_r = 0.0, T_l = 0.0, T_d = 0.0, T_u = 0.0;
  double exT_d = 0.0, exT_u = 0.0;
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

      if (j + 1 <= nx - 1) {
        T_r = (coef(i, j) + coef(i, j + 1)) / 2.0;
        tripletList.push_back(T(idRow, ij2ind(i, j + 1), -T_r));
      }

      if (j - 1 >= 0) {
        T_l = (coef(i, j) + coef(i, j - 1)) / 2.0;
        tripletList.push_back(T(idRow, ij2ind(i, j - 1), -T_l));
      }

      if (i + 1 <= nz - 1) {
        T_d = (coef(i, j) + coef(i + 1, j)) / 2.0;
        tripletList.push_back(T(idRow, ij2ind(i + 1, j), -T_d));
      } else {
        exT_d = 1.5 * coef(i, j) - 0.5 * coef(i - 1, j);
      }

      if (i - 1 >= 0) {
        T_u = (coef(i, j) + coef(i - 1, j)) / 2.0;
        tripletList.push_back(T(idRow, ij2ind(i - 1, j), -T_u));
      } else {
        exT_u = 1.5 * coef(i, j) - 0.5 * coef(i + 1, j);
      }

      tripletList.push_back(T(idRow, ij2ind(i, j), T_r + T_l + T_d + T_u));

      rhs(idRow) = h2 * g(i, j) - exT_u * rhograv * h + exT_d * rhograv * h;
    }
  }
  tripletList.push_back(T(0, ij2ind(0, 0), 1.0));
  // tripletList.push_back(T(0, ij2ind(0, 0), coef(0, 0)));

  Amat.setFromTriplets(tripletList.begin(), tripletList.end());  // row-major
}

#endif