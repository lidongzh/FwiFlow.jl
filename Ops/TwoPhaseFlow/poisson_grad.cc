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

REGISTER_OP("PoissonGrad")
.Input("grad: double")
.Input("coef: double")
.Input("g: double")
.Input("h: double")
.Output("grad_coef: double")
.Output("grad_g: double")
.Output("grad_h: double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
  shape_inference::ShapeHandle grad_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &grad_shape));
  shape_inference::ShapeHandle coef_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &coef_shape));
  shape_inference::ShapeHandle g_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &g_shape));
  shape_inference::ShapeHandle h_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));

  c->set_output(0, c->Matrix(c->Dim(coef_shape, 0), c->Dim(coef_shape, 1)));
  c->set_output(1, c->Matrix(c->Dim(g_shape, 0), c->Dim(g_shape, 1)));
  c->set_output(2, c->Scalar());
  return Status::OK();
});

/// \brief Implementation of an inner product operation.
/// \param context
/// \author David Stutz
class PoissonGradOp : public OpKernel {
 public:
  /// \brief Constructor.
  /// \param context
  explicit PoissonGradOp(OpKernelConstruction *context) : OpKernel(context) {
  }

  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext *context) override {
    // some checks to be sure ...
    DCHECK_EQ(4, context->num_inputs());

    // get the input tensor
    const Tensor &grad = context->input(0);
    const Tensor &coef = context->input(1);
    const Tensor &g = context->input(2);
    const Tensor &h = context->input(3);

    // check shapes of input and weights
    const TensorShape &grad_shape = grad.shape();
    const TensorShape &coef_shape = coef.shape();
    const TensorShape &g_shape = g.shape();
    const TensorShape &h_shape = h.shape();

    // check input is a standing vector
    DCHECK_EQ(coef_shape.dims(), 2);
    int m = coef_shape.dim_size(0), n = coef_shape.dim_size(1);
    DCHECK_EQ(grad.dims(), 2);
    DCHECK_EQ(grad.dim_size(0), m); DCHECK_EQ(grad.dim_size(1), n);

    DCHECK_EQ(g_shape.dims(), 2);
    DCHECK_EQ(g_shape.dim_size(0), m); DCHECK_EQ(g_shape.dim_size(1), n);

    DCHECK_EQ(h_shape.dims(), 0);

    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(m);
    output_shape.AddDim(n);

    // create output tensor
    Tensor *grad_coef = NULL, *grad_g = NULL, *grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &grad_coef));
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &grad_g));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape(), &grad_h));

    // get the corresponding Eigen tensors for data access
    auto grad_tensor = grad.matrix<double>();
    auto coef_tensor = coef.matrix<double>();
    auto g_tensor = g.matrix<double>();
    auto h_tensor = h.scalar<double>();

    auto grad_coef_tensor = grad_coef->matrix<double>();
    auto grad_g_tensor = grad_g->matrix<double>();
    auto grad_h_tensor = grad_h->scalar<double>();

    // // // copy data to eigen3
    // densMat grad_mat(m, n);
    // densMat u_mat(m, n);
    // densMat a_mat(m, n);
    // densMat b1_mat(m, n);
    // densMat b2_mat(m, n);
    // densMat du_mat(m, n);
    // densMat da_mat(m, n);
    // densMat db1_mat(m, n);
    // densMat db2_mat(m, n);
    // for (int i = 0; i < m; i++) {
    //   for (int j = 0; j < n; j++) {
    //     if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
    //       du_mat(i, j) = 0.0; da_mat(i, j) = 0.0; db1_mat(i, j) = 0.0; db2_mat(i, j) = 0.0;
    //     }
    //     u_mat(i, j) = u_tensor(i, j);
    //     a_mat(i, j) = a_tensor(i, j);
    //     b1_mat(i, j) = b1_tensor(i, j);
    //     b2_mat(i, j) = b2_tensor(i, j);
    //     grad_mat(i, j) = grad_tensor(i, j);
    //   }
    // }
    // backward(grad_mat, u_mat, a_mat, b1_mat, b2_mat, du_mat, da_mat, db1_mat, db2_mat, dt, h);
    // for (int i = 0; i < m; i++) {
    //   for (int j = 0; j < n; j++) {
    //     du_tensor(i, j) = du_mat(i, j);
    //     da_tensor(i, j) = da_mat(i, j);
    //     db1_tensor(i, j) = db1_mat(i, j);
    //     db2_tensor(i, j) = db2_mat(i, j);
    //   }
    // }
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        grad_coef_tensor(i, j) = coef_tensor(i, j);
        grad_g_tensor(i, j) = g_tensor(i, j);
      }
    }
    grad_h_tensor() = h_tensor();

  }
};

REGISTER_KERNEL_BUILDER(Name("PoissonGrad").Device(DEVICE_CPU), PoissonGradOp);