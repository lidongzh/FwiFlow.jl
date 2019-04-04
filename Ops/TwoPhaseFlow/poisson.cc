/// \file inner_product.cc
/// \author David Stutz
/// \brief Implementation of a inner product (i.e. fully connected layer)
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "poisson.h"
using namespace tensorflow;

REGISTER_OP("Poisson")
.Input("coef: double")
.Input("g: double")
.Input("h: double")
.Output("p: double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
  shape_inference::ShapeHandle coef_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &coef_shape));
  shape_inference::ShapeHandle g_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &g_shape));
  shape_inference::ShapeHandle h_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &h_shape));

  c->set_output(0, c->Matrix(c->Dim(coef_shape, 0), c->Dim(coef_shape, 1)));
  return Status::OK();
});

/// \brief Implementation of an inner product operation.
/// \param context
/// \author David Stutz
class PoissonOp : public OpKernel {
 public:
  /// \brief Constructor.
  /// \param context
  explicit PoissonOp(OpKernelConstruction *context) : OpKernel(context) {
  }

  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext *context) override {

    // some checks to be sure ...
    DCHECK_EQ(3, context->num_inputs());

    // get the input tensor
    const Tensor &coef = context->input(0);
    const Tensor &g = context->input(1);
    const Tensor &h = context->input(2);

    // check shapes of input and weights
    const TensorShape &coef_shape = coef.shape();
    const TensorShape &g_shape = g.shape();
    const TensorShape &h_shape = h.shape();

    // check input is a standing vector
    DCHECK_EQ(coef_shape.dims(), 2);
    int nz = coef_shape.dim_size(0), nx = coef_shape.dim_size(1);
    DCHECK_EQ(g_shape.dims(), 2);
    DCHECK_EQ(g_shape.dim_size(0), nz); DCHECK_EQ(g_shape.dim_size(1), nx);

    DCHECK_EQ(h_shape.dims(), 0);


    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(nz);
    output_shape.AddDim(nx);

    // create output tensor
    Tensor *p = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &p));

    // get the corresponding Eigen tensors for data access
    auto coef_tensor = coef.matrix<double>();
    auto g_tensor = g.matrix<double>();
    auto h_tensor = h.scalar<double>();
    double dh = h_tensor();
    double dh2 = pow(dh, 2);
    auto p_tensor = p->matrix<double>();

    // assemble matrix
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(5);
    Eigen::VectorXd rhs(nz * nx);
    int idRow = 0;
    double a = 1.0, b = 1.0, c = 1.0, d = 1.0;
    for (int j = 0; j < nx; j++) {
      for (int i = 0; i < nz; i++) {
        idRow = j * nz + i;
        if (i == 0 && j >= 1 && j <= nx - 2) {
          // upper BD
          tripletList.push_back(T(idRow, j * nz + i, a + b + c));
          tripletList.push_back(T(idRow, nz * (j + 1) + i, -a));
          tripletList.push_back(T(idRow, nz * (j - 1) + i, -b));
          tripletList.push_back(T(idRow, j * nz + i + 1, -c));
          rhs(idRow) = dh2 * g_tensor(i, j) - d * dh * RHO_o * GRAV;

        } else if (i == nz - 1, j >= 1 && j <= nx - 2) {
          // upper BD
          tripletList.push_back(T(idRow, j * nz + i, a + b + d));
          tripletList.push_back(T(idRow, nz * (j + 1) + i, -a));
          tripletList.push_back(T(idRow, nz * (j - 1) + i, -b));
          tripletList.push_back(T(idRow, j * nz + i - 1, -d));
          rhs(idRow) = dh2 * g_tensor(i, j) + c * dh * RHO_o * GRAV;

        } else if (j == 0 && i >= 1 && i <= nz - 2) {
          // left BD
          tripletList.push_back(T(idRow, j * nz + i, a + c + d));
          tripletList.push_back(T(idRow, nz * (j + 1) + i, -a));
          tripletList.push_back(T(idRow, j * nz + i + 1, -c));
          tripletList.push_back(T(idRow, j * nz + i - 1, -d));
          rhs(idRow) = dh2 * g_tensor(i, j);

        } else if (j == nx - 1 && i >= 1 && i <= nz - 2) {
          // right BD
          tripletList.push_back(T(idRow, j * nz + i, b + c + d));
          tripletList.push_back(T(idRow, nz * (j - 1) + i, -b));
          tripletList.push_back(T(idRow, j * nz + i + 1, -c));
          tripletList.push_back(T(idRow, j * nz + i - 1, -d));
          rhs(idRow) = dh2 * g_tensor(i, j);

        } else if (i == 0 && j == 0) {
          // upper-left corner
          tripletList.push_back(T(idRow, nz * (j + 1) + i, -a));
          tripletList.push_back(T(idRow, j * nz + i + 1, -c));
          rhs(idRow) = dh2 * g_tensor(i, j) - d * dh * RHO_o * GRAV;
        } else if (i == 0 && j == nx - 1) {
          // upper-right corner
          tripletList.push_back(T(idRow, j * nz + i, b + c));
          tripletList.push_back(T(idRow, nz * (j - 1) + i, -b));
          tripletList.push_back(T(idRow, j * nz + i + 1, -c));
          rhs(idRow) = dh2 * g_tensor(i, j) - d * dh * RHO_o * GRAV;
        } else if (i == nz - 1 && j == 0) {
          // lower-left corner
          tripletList.push_back(T(idRow, j * nz + i, a + d));
          tripletList.push_back(T(idRow, nz * (j + 1) + i, -a));
          tripletList.push_back(T(idRow, j * nz + i - 1, -d));
          rhs(idRow) = dh2 * g_tensor(i, j) + c * dh * RHO_o * GRAV;
        } else if (i == nz - 1 && j == nx - 1) {
          // lower-right corner
          tripletList.push_back(T(idRow, j * nz + i, b + d));
          tripletList.push_back(T(idRow, nz * (j - 1) + i, -b));
          tripletList.push_back(T(idRow, j * nz + i - 1, -d));
          rhs(idRow) = dh2 * g_tensor(i, j) + c * dh * RHO_o * GRAV;
        } else {
          tripletList.push_back(T(idRow, j * nz + i, a + b + c + d));
          tripletList.push_back(T(idRow, nz * (j + 1) + i, -a));
          tripletList.push_back(T(idRow, nz * (j - 1) + i, -b));
          tripletList.push_back(T(idRow, j * nz + i + 1, -c));
          tripletList.push_back(T(idRow, j * nz + i - 1, -d));
          rhs(idRow) = dh2 * g_tensor(i, j);
        }
      }
    }
    SparseMatrixType Amat(nz * nx, nz * nx);
    Amat.setFromTriplets(tripletList.begin(), tripletList.end());


    // // copy data to eigen3
    // densMat u_mat(m, n);
    // densMat a_mat(m, n);
    // densMat b1_mat(m, n);
    // densMat b2_mat(m, n);
    // densMat y_mat(m, n);
    // for (int i = 0; i < m; i++) {
    //   for (int j = 0; j < n; j++) {
    //     if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
    //       u_mat(i, j) = 0.0; a_mat(i, j) = 0.0; b1_mat(i, j) = 0.0; b2_mat(i, j) = 0.0;
    //     }
    //     u_mat(i, j) = u_tensor(i, j);
    //     a_mat(i, j) = a_tensor(i, j);
    //     b1_mat(i, j) = b1_tensor(i, j);
    //     b2_mat(i, j) = b2_tensor(i, j);
    //     y_mat(i, j) = 0.0;
    //   }
    // }
    // forward(u_mat, a_mat, b1_mat, b2_mat, y_mat, dt, h);
    // for (int i = 0; i < m; i++) {
    //   for (int j = 0; j < n; j++) {
    //     y_tensor(i, j) = y_mat(i, j);
    //   }
    // }
    for (int j = 0; j < nz; j++) {
      for (int i = 0; i < nx; i++) {
        p_tensor(j, i) = coef_tensor(j, i);
      }
    }

  }
};

REGISTER_KERNEL_BUILDER(Name("Poisson").Device(DEVICE_CPU), PoissonOp);