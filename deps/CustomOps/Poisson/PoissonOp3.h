// Dongzhuo Li 04/01/2019
#ifndef POISSONOP_H__
#define POISSONOP_H__

#define RHO_o 1000.0
#define GRAV 9.8
// double array row major
// double array uninitialized
#define coef(z,x)  		coef[(z)*(nx)+(x)] // row major
#define g(z,x)  			g[(z)*(nx)+(x)] // row major
#define ij2ind(z,x)			((z)*(nx)+(x))

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>

void forward(double *p, const double *coef, const double *g, double h, int nz, int nx) {
	// assemble matrix
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(5);
	// Eigen::SparseMatrix<double, Eigen::RowMajor> Amat(nz * nx + 1, nz * nx);
	Eigen::SparseMatrix<double, Eigen::RowMajor> Amat(nz * nx, nz * nx);
	Eigen::VectorXd rhs(nz * nx), pvec(nz * nx);
	int idRow = 0;
	double T_r = 0.0, T_l = 0.0, T_d = 0.0, T_u = 0.0;
	double exT_d = 0.0, exT_u = 0.0;
	double h2 = h * h;
	for (int i = 0; i < nz; i++) {
		for (int j = 0; j < nx; j++) {
			idRow = i * nx + j; // row-major
			T_r = 0.0; T_l = 0.0; T_d = 0.0; T_u = 0.0;
			exT_d = 0.0; exT_u = 0.0;

			if (j + 1 <= nx - 1) {
				// T_r = coef(i, j) * coef(i, j + 1) / (coef(i, j) + coef(i, j + 1));
				T_r = (coef(i, j) + coef(i, j + 1)) / 2.0;
				std::cout << "T_r = " << (coef(i, j) * coef(i, j + 1)) / 2.0 << std::endl;
				tripletList.push_back(T(idRow, ij2ind(i, j + 1), -T_r));
			}

			if (j - 1 >= 0) {
				// T_l = coef(i, j) * coef(i, j - 1) / (coef(i, j) + coef(i, j - 1));
				T_l = (coef(i, j) + coef(i, j - 1)) / 2.0;
				tripletList.push_back(T(idRow, ij2ind(i, j - 1), -T_l));
			}

			if (i + 1 <= nz - 1) {
				// T_d = coef(i, j) * coef(i + 1, j) / (coef(i, j) + coef(i + 1, j));
				T_d = (coef(i, j) + coef(i + 1, j)) / 2.0;
				tripletList.push_back(T(idRow, ij2ind(i + 1, j), -T_d));
			} else {
				// exT_d = (2.0 * coef(i, j) - coef(i + 1, j)) * coef(i, j) / ((2.0 * coef(i, j) - coef(i + 1, j)) + coef(i, j));
				exT_d = 1.5 * coef(i, j) - 0.5 * coef(i - 1, j);
			}

			if (i - 1 >= 0) {
				// T_u = coef(i, j) * coef(i - 1, j) / (coef(i, j) + coef(i - 1, j));
				T_u = (coef(i, j) + coef(i - 1, j)) / 2.0;
				tripletList.push_back(T(idRow, ij2ind(i - 1, j), -T_u));
			} else {
				// exT_u = (2.0 * coef(i, j) - coef(i - 1, j)) * coef(i, j) / ((2.0 * coef(i, j) - coef(i - 1, j)) + coef(i, j));
				exT_u = 1.5 * coef(i, j) - 0.5 * coef(i + 1, j);
			}

			tripletList.push_back(T(idRow, ij2ind(i, j), T_r + T_l + T_d + T_u));

			rhs(idRow) = h2 * g(i, j) - exT_u * RHO_o * GRAV * h + exT_d * RHO_o * GRAV * h;
		}
	}
	tripletList.push_back(T(0, ij2ind(0, 0), coef(0, 0)));

	std::cout << "mean(rhs) = " << rhs.mean() << std::endl;
	std::cout << "sum(rhs) = " << rhs.sum() << std::endl;
// rhs = rhs.array() - rhs.mean();
	std::cout << "sum(rhs) = " << rhs.sum() << std::endl;

	Amat.setFromTriplets(tripletList.begin(), tripletList.end()); // row-major

	Eigen::MatrixXd dMat = Eigen::MatrixXd(Amat);
	std::cout << "Amat = " << dMat << std::endl;
// std::cout << "Amat is compressed: ?" << Amat.isCompressed() << std::endl;
	Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;

// Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>  solver;
	solver.analyzePattern(Amat);
// solver.preconditioner().setDroptol(0.001);
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
	// pvec = pvec.array() - pvec.mean();

	for (int i = 0; i < nz; i++) {
		for (int j = 0; j < nx; j++) {
			p[ij2ind(i, j)] = pvec(ij2ind(i, j));
		}
	}

	std::cout << "mean(p) = " << pvec.mean() << std::endl;

}


void backward(const double *grad_p, const double *p, const double *coef, const double *g, double h, int nz, int nx,
              double *grad_coef, double *grad_g) {
	// F_p * p_v + F_v = 0
	// J_v = J_p * p_v = -J_p * F^{-1}_p * F_v

}




#endif