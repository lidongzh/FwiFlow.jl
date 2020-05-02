// Dongzhuo Li 04/07/2019
#ifndef LAPLACIAN_H__
#define LAPLACIAN_H__

// double array row major
// double array uninitialized
#define coef(z,x)  		coef[(z)*(nx)+(x)] // row major
#define func(z,x)  			func[(z)*(nx)+(x)] // row major
#define ij2ind(z,x)			((z)*(nx)+(x))
#define out(z,x)				out[(z)*(nx)+(x)]
#define grad_out(z,x)		grad_out[(z)*(nx)+(x)]
#define grad_coef(z,x)		grad_coef[(z)*(nx)+(x)]
#define grad_func(z,x)		grad_func[(z)*(nx)+(x)]

void intialArray(double *ip, int size, double value) {
	for (int i = 0; i < size; i++) {
		ip[i] = value;
		// printf("value = %f\n", value);
	}
}

void forward(double *out, const double *coef, const double *func, double h, double rhograv, int nz, int nx) {
	double T_r = 0.0, T_l = 0.0, T_d = 0.0, T_u = 0.0;
	double F_r = 0.0, F_l = 0.0, F_d = 0.0, F_u = 0.0;
	for (int i = 0; i < nz; i++) {
		for (int j = 0; j < nx; j++) {
			T_r = 0.0; T_l = 0.0; T_d = 0.0; T_u = 0.0;
			F_r = 0.0; F_l = 0.0; F_d = 0.0; F_u = 0.0;

			if (j + 1 <= nx - 1) {
				T_r = (coef(i, j + 1) + coef(i, j)) / 2.0;
				F_r = (func(i, j + 1) - func(i, j)) / h;
			}

			if (j - 1 >= 0) {
				T_l = (coef(i, j) + coef(i, j - 1)) / 2.0;
				F_l = (func(i, j) - func(i, j - 1)) / h;
			}

			if (i + 1 <= nz - 1) {
				T_d = (coef(i + 1, j) + coef(i, j)) / 2.0;
				F_d = (func(i + 1, j) - func(i, j)) / h;
			} else {
				F_d = rhograv;
				T_d = 1.5 * coef(i, j) - 0.5 * coef(i - 1, j);
			}

			if (i - 1 >= 0) {
				T_u = (coef(i, j) + coef(i - 1, j)) / 2.0;
				F_u = (func(i, j) - func(i - 1, j)) / h;
			} else {
				F_u = rhograv;
				T_u = 1.5 * coef(i, j) - 0.5 * coef(i + 1, j);
			}

			out(i, j) = (T_d * F_d - T_u * F_u + T_r * F_r - T_l * F_l) / h;

		}
	}
}



void backward(const double *grad_out, const double *coef, \
              const double *func, double h, double rhograv, double *grad_coef, double *grad_func, int nz, int nx) {

	intialArray(grad_coef, nz * nx, 0.0);
	intialArray(grad_func, nz * nx, 0.0);
	double T_r = 0.0, T_l = 0.0, T_d = 0.0, T_u = 0.0;
	double F_r = 0.0, F_l = 0.0, F_d = 0.0, F_u = 0.0;

	for (int i = 0; i < nz; i++) {
		for (int j = 0; j < nx; j++) {
			T_r = 0.0; T_l = 0.0; T_d = 0.0; T_u = 0.0;
			F_r = 0.0; F_l = 0.0; F_d = 0.0; F_u = 0.0;

			if (j + 1 <= nx - 1) {
				T_r = (coef(i, j + 1) + coef(i, j)) / 2.0;
				F_r = (func(i, j + 1) - func(i, j)) / h;
				grad_func(i, j) += -T_r / (h * h) * grad_out(i, j);
				grad_coef(i, j) += F_r / h * 0.5 * grad_out(i, j);
				grad_func(i, j + 1) += T_r / (h * h) * grad_out(i, j);
				grad_coef(i, j + 1) += F_r / h * 0.5 * grad_out(i, j);
			}

			if (j - 1 >= 0) {
				T_l = (coef(i, j) + coef(i, j - 1)) / 2.0;
				F_l = (func(i, j) - func(i, j - 1)) / h;
				grad_func(i, j) += -T_l / (h * h) * grad_out(i, j);
				grad_coef(i, j) += -F_l / h * 0.5 * grad_out(i, j);
				grad_func(i, j - 1) += T_l / (h * h) * grad_out(i, j);
				grad_coef(i, j - 1) += -F_l / h * 0.5 * grad_out(i, j);
			}

			if (i + 1 <= nz - 1) {
				T_d = (coef(i + 1, j) + coef(i, j)) / 2.0;
				F_d = (func(i + 1, j) - func(i, j)) / h;
				grad_func(i, j) += -T_d / (h * h) * grad_out(i, j);
				grad_coef(i, j) += F_d / h * 0.5 * grad_out(i, j);
				grad_func(i + 1, j) += T_d / (h * h) * grad_out(i, j);
				grad_coef(i + 1, j) += F_d / h * 0.5 * grad_out(i, j);
			} else {
				F_d = rhograv;
				T_d = 1.5 * coef(i, j) - 0.5 * coef(i - 1, j);
				grad_coef(i, j) += F_d / h * 1.5 * grad_out(i, j);
				grad_coef(i, j) += -F_d / h * 0.5 * grad_out(i, j);
			}

			if (i - 1 >= 0) {
				T_u = (coef(i, j) + coef(i - 1, j)) / 2.0;
				F_u = (func(i, j) - func(i - 1, j)) / h;
				grad_func(i, j) += -T_u / (h * h) * grad_out(i, j);
				grad_coef(i, j) += -F_u / h * 0.5 * grad_out(i, j);
				grad_func(i - 1, j) += T_u / (h * h) * grad_out(i, j);
				grad_coef(i - 1, j) += -F_u / h * 0.5 * grad_out(i, j);
			} else {
				F_u = rhograv;
				T_u = 1.5 * coef(i, j) - 0.5 * coef(i + 1, j);
				grad_coef(i, j) += -F_u / h * 1.5 * grad_out(i, j);
				grad_coef(i, j) += F_u / h * 0.5 * grad_out(i, j);
			}

		}
	}
}

#endif