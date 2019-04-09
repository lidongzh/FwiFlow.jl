#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <unsupported/Eigen/SparseExtra> // For reading MatrixMarket files

#include <amgcl/backend/eigen.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mm>" << std::endl;
        return 1;
    }

    // Read sparse matrix from MatrixMarket format.
    // In general this should come pre-assembled.
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
    Eigen::loadMarket(A, argv[1]);

    // Use vector of ones as RHS for simplicity:
    Eigen::VectorXd f = Eigen::VectorXd::Constant(A.rows(), 1.0);

    // Zero initial approximation:
    Eigen::VectorXd x = Eigen::VectorXd::Zero(A.rows());

    // Setup the solver:
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::eigen<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::bicgstab<amgcl::backend::eigen<double> >
        > Solver;

    Solver solve(A);
    std::cout << solve << std::endl;

    // Solve the system for the given RHS:
    int    iters;
    double error;
    std::tie(iters, error) = solve(f, x);

    std::cout << iters << " " << error << std::endl;
}