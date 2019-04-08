#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
using namespace tensorflow;
#include "Laplacian.h"

REGISTER_OP("Laplacian")

.Input("coef : double")
.Input("func : double")
.Input("h : double")
.Input("rhograv : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {

  shape_inference::ShapeHandle coef_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &coef_shape));
  shape_inference::ShapeHandle func_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &func_shape));
  shape_inference::ShapeHandle h_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &h_shape));
  shape_inference::ShapeHandle rhograv_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &rhograv_shape));

  c->set_output(0, c->Matrix(-1, -1));
  return Status::OK();
});
class LaplacianOp : public OpKernel {
 private:

 public:
  explicit LaplacianOp(OpKernelConstruction *context) : OpKernel(context) {

  }

  void Compute(OpKernelContext *context) override {
    DCHECK_EQ(4, context->num_inputs());


    const Tensor &coef = context->input(0);
    const Tensor &func = context->input(1);
    const Tensor &h = context->input(2);
    const Tensor &rhograv = context->input(3);


    const TensorShape &coef_shape = coef.shape();
    const TensorShape &func_shape = func.shape();
    const TensorShape &h_shape = h.shape();
    const TensorShape &rhograv_shape = rhograv.shape();


    DCHECK_EQ(coef_shape.dims(), 2);
    DCHECK_EQ(func_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(rhograv_shape.dims(), 0);

    // extra check

    // create output shape
    int nz = coef_shape.dim_size(0), nx = coef_shape.dim_size(1);

    TensorShape out_shape({nz, nx});

    // create output tensor

    Tensor *out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));

    // get the corresponding Eigen tensors for data access

    auto coef_tensor = coef.flat<double>().data();
    auto func_tensor = func.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto rhograv_tensor = rhograv.flat<double>().data();
    auto out_tensor = out->flat<double>().data();

    // implement your forward function here

    // TODO:
    forward(out_tensor, coef_tensor, func_tensor, *h_tensor, *rhograv_tensor, nz, nx);

  }
};
REGISTER_KERNEL_BUILDER(Name("Laplacian").Device(DEVICE_CPU), LaplacianOp);


REGISTER_OP("LaplacianGrad")

.Input("grad_out : double")
.Input("coef : double")
.Input("func : double")
.Input("h : double")
.Input("rhograv : double")
.Output("grad_coef : double")
.Output("grad_func : double")
.Output("grad_h : double")
.Output("grad_rhograv : double");
class LaplacianGradOp : public OpKernel {
 private:

 public:
  explicit LaplacianGradOp(OpKernelConstruction *context) : OpKernel(context) {

  }

  void Compute(OpKernelContext *context) override {


    const Tensor &grad_out = context->input(0);
    const Tensor &coef = context->input(1);
    const Tensor &func = context->input(2);
    const Tensor &h = context->input(3);
    const Tensor &rhograv = context->input(4);


    const TensorShape &coef_shape = coef.shape();
    const TensorShape &func_shape = func.shape();
    const TensorShape &h_shape = h.shape();
    const TensorShape &rhograv_shape = rhograv.shape();
    const TensorShape &grad_out_shape = grad_out.shape();


    DCHECK_EQ(coef_shape.dims(), 2);
    DCHECK_EQ(func_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(rhograv_shape.dims(), 0);
    DCHECK_EQ(grad_out_shape.dims(), 2);

    // extra check

    // create output shape

    TensorShape grad_coef_shape(coef_shape);
    TensorShape grad_func_shape(func_shape);
    TensorShape grad_h_shape({});
    TensorShape grad_rhograv_shape({});

    // create output tensor
    int nz = coef_shape.dim_size(0), nx = coef_shape.dim_size(1);

    Tensor *grad_coef = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_coef_shape, &grad_coef));
    Tensor *grad_func = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_func_shape, &grad_func));
    Tensor *grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_h_shape, &grad_h));
    Tensor *grad_rhograv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_rhograv_shape, &grad_rhograv));

    // get the corresponding Eigen tensors for data access

    auto coef_tensor = coef.flat<double>().data();
    auto func_tensor = func.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto rhograv_tensor = rhograv.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto grad_coef_tensor = grad_coef->flat<double>().data();
    auto grad_func_tensor = grad_func->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();
    auto grad_rhograv_tensor = grad_rhograv->flat<double>().data();

    // implement your backward function here

    // TODO:
    backward(grad_out_tensor, coef_tensor, \
             func_tensor, *h_tensor, *rhograv_tensor, grad_coef_tensor, grad_func_tensor, nz, nx);

  }
};
REGISTER_KERNEL_BUILDER(Name("LaplacianGrad").Device(DEVICE_CPU), LaplacianGradOp);

