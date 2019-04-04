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
	double a = -1.0, b = -1.0, c = -1.0, d = -1.0;
	double h2 = h * h;
	for (int i = 0; i < nz; i++) {
		for (int j = 0; j < nx; j++) {
			idRow = i * nx + j; // row-major
			if (i == 0 && j >= 1 && j <= nx - 2) {
				// upper BD
				a = (coef(i, j) + coef(i, j + 1)) / 2.0;
				b = (coef(i, j) + coef(i, j - 1)) / 2.0;
				c = (coef(i, j) + coef(i + 1, j)) / 2.0;
				d = 1.5 * coef(i, j) - 0.5 * coef(i + 1, j);
				tripletList.push_back(T(idRow, ij2ind(i, j), a + b + c));
				tripletList.push_back(T(idRow, ij2ind(i, j + 1), -a));
				tripletList.push_back(T(idRow, ij2ind(i, j - 1), -b));
				tripletList.push_back(T(idRow, ij2ind(i + 1, j), -c));
				rhs(idRow) = h2 * g(i, j) - d * h * RHO_o * GRAV;

			} else if (i == nz - 1, j >= 1 && j <= nx - 2) {
				// lower BD
				a = (coef(i, j) + coef(i, j + 1)) / 2.0;
				b = (coef(i, j) + coef(i, j - 1)) / 2.0;
				c = 1.5 * coef(i, j) - 0.5 * coef(i - 1, j);
				d = (coef(i, j) + coef(i - 1, j)) / 2.0;
				tripletList.push_back(T(idRow, ij2ind(i, j), a + b + d));
				tripletList.push_back(T(idRow, ij2ind(i, j + 1), -a));
				tripletList.push_back(T(idRow, ij2ind(i, j - 1), -b));
				tripletList.push_back(T(idRow, ij2ind(i - 1, j), -d));
				rhs(idRow) = h2 * g(i, j) + c * h * RHO_o * GRAV;

			} else if (j == 0 && i >= 1 && i <= nz - 2) {
				// left BD
				a = (coef(i, j) + coef(i, j + 1)) / 2.0;
				b = 1.5 * coef(i, j) - 0.5 * coef(i, j + 1);
				c = (coef(i, j) + coef(i + 1, j)) / 2.0;
				d = (coef(i, j) + coef(i - 1, j)) / 2.0;
				tripletList.push_back(T(idRow, ij2ind(i, j), a + c + d));
				tripletList.push_back(T(idRow, ij2ind(i, j + 1), -a));
				tripletList.push_back(T(idRow, ij2ind(i + 1, j), -c));
				tripletList.push_back(T(idRow, ij2ind(i - 1, j), -d));
				rhs(idRow) = h2 * g(i, j);

			} else if (j == nx - 1 && i >= 1 && i <= nz - 2) {
				// right BD
				a = 1.5 * coef(i, j) - 0.5 * coef(i, j - 1);
				b = (coef(i, j) + coef(i, j - 1)) / 2.0;
				c = (coef(i, j) + coef(i + 1, j)) / 2.0;
				d = (coef(i, j) + coef(i - 1, j)) / 2.0;
				tripletList.push_back(T(idRow, ij2ind(i, j), b + c + d));
				tripletList.push_back(T(idRow, ij2ind(i, j - 1), -b));
				tripletList.push_back(T(idRow, ij2ind(i + 1, j), -c));
				tripletList.push_back(T(idRow, ij2ind(i - 1, j), -d));
				rhs(idRow) = h2 * g(i, j);

			} else if (i == 0 && j == 0) {
				// upper-left corner
				a = (coef(i, j) + coef(i, j + 1)) / 2.0;
				b = 1.5 * coef(i, j) - 0.5 * coef(i, j + 1);
				c = (coef(i, j) + coef(i + 1, j)) / 2.0;
				d = 1.5 * coef(i, j) - 0.5 * coef(i + 1, j);
				tripletList.push_back(T(idRow, ij2ind(i, j), a + c));
				tripletList.push_back(T(idRow, ij2ind(i, j + 1), -a));
				tripletList.push_back(T(idRow, ij2ind(i + 1, j), -c));
				rhs(idRow) = h2 * g(i, j) - d * h * RHO_o * GRAV;
				tripletList.push_back(T(idRow, ij2ind(i, j), coef(0, 0)));

			} else if (i == 0 && j == nx - 1) {
				// upper-right corner
				a = 1.5 * coef(i, j) - 0.5 * coef(i, j - 1);
				b = (coef(i, j) + coef(i, j - 1)) / 2.0;
				c = (coef(i, j) + coef(i + 1, j)) / 2.0;
				d = 1.5 * coef(i, j) - 0.5 * coef(i + 1, j);
				tripletList.push_back(T(idRow, ij2ind(i, j), b + c));
				tripletList.push_back(T(idRow, ij2ind(i, j - 1), -b));
				tripletList.push_back(T(idRow, ij2ind(i + 1, j), -c));
				rhs(idRow) = h2 * g(i, j) - d * h * RHO_o * GRAV;

			} else if (i == nz - 1 && j == 0) {
				// lower-left corner
				a = (coef(i, j) + coef(i, j + 1)) / 2.0;
				b = 1.5 * coef(i, j) - 0.5 * coef(i, j + 1);
				c = 1.5 * coef(i, j) - 0.5 * coef(i - 1, j);
				d = (coef(i, j) + coef(i - 1, j)) / 2.0;
				tripletList.push_back(T(idRow, ij2ind(i, j), a + d));
				tripletList.push_back(T(idRow, ij2ind(i, j + 1), -a));
				tripletList.push_back(T(idRow, ij2ind(i - 1, j), -d));
				rhs(idRow) = h2 * g(i, j) + c * h * RHO_o * GRAV;

			} else if (i == nz - 1 && j == nx - 1) {
				// lower-right corner
				a = 1.5 * coef(i, j) - 0.5 * coef(i, j - 1);
				b = (coef(i, j) + coef(i, j - 1)) / 2.0;
				c = 1.5 * coef(i, j) - 0.5 * coef(i - 1, j);
				d = (coef(i, j) + coef(i - 1, j)) / 2.0;
				tripletList.push_back(T(idRow, ij2ind(i, j), b + d));
				tripletList.push_back(T(idRow, ij2ind(i, j - 1), -b));
				tripletList.push_back(T(idRow, ij2ind(i - 1, j), -d));
				rhs(idRow) = h2 * g(i, j) + c * h * RHO_o * GRAV;

			} else {
				a = (coef(i, j) + coef(i, j + 1)) / 2.0;
				b = (coef(i, j) + coef(i, j - 1)) / 2.0;
				c = (coef(i, j) + coef(i + 1, j)) / 2.0;
				d = (coef(i, j) + coef(i - 1, j)) / 2.0;
				tripletList.push_back(T(idRow, ij2ind(i, j), a + b + c + d));
				tripletList.push_back(T(idRow, ij2ind(i, j + 1), -a));
				tripletList.push_back(T(idRow, ij2ind(i, j - 1), -b));
				tripletList.push_back(T(idRow, ij2ind(i + 1, j), -c));
				tripletList.push_back(T(idRow, ij2ind(i - 1, j), -d));
				rhs(idRow) = h2 * g(i, j);
			}
			std::cout << "T_l = " << b << ", T_r = " << a << ", T_u = " << d << ", T_d = " << c << std::endl;
		}
	}

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