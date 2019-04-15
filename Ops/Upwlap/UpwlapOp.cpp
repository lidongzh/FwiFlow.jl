#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
using namespace tensorflow;
// #include "ADEL.h"
#include "UpwlapOp.h"

REGISTER_OP("UpwlapOp")

.Input("perm : double")
.Input("mobi : double")
.Input("func : double")
.Input("h : double")
.Input("rhograv : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

  shape_inference::ShapeHandle perm_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &perm_shape));
  shape_inference::ShapeHandle mobi_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &mobi_shape));
  shape_inference::ShapeHandle func_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &func_shape));
  shape_inference::ShapeHandle h_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));
  shape_inference::ShapeHandle rhograv_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &rhograv_shape));
  c->set_output(0, c->Matrix(c->Dim(c->input(0),0), c->Dim(c->input(0),1)));
  return Status::OK();
});
class UpwlapOpOp : public OpKernel {
 private:

 public:
  explicit UpwlapOpOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {
    DCHECK_EQ(5, context->num_inputs());


    const Tensor& perm = context->input(0);
    const Tensor& mobi = context->input(1);
    const Tensor& func = context->input(2);
    const Tensor& h = context->input(3);
    const Tensor& rhograv = context->input(4);


    const TensorShape& perm_shape = perm.shape();
    const TensorShape& mobi_shape = mobi.shape();
    const TensorShape& func_shape = func.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& rhograv_shape = rhograv.shape();


    DCHECK_EQ(perm_shape.dims(), 2);
    DCHECK_EQ(mobi_shape.dims(), 2);
    DCHECK_EQ(func_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(rhograv_shape.dims(), 0);

    // extra check

    // create output shape
    int nz = func_shape.dim_size(0), nx = func_shape.dim_size(1);

    TensorShape out_shape({nz, nx});

    // create output tensor

    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));

    // get the corresponding Eigen tensors for data access

    auto perm_tensor = perm.flat<double>().data();
    auto mobi_tensor = mobi.flat<double>().data();
    auto func_tensor = func.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto rhograv_tensor = rhograv.flat<double>().data();
    auto out_tensor = out->flat<double>().data();

    // implement your forward function here

    // TODO:
    forward(out_tensor, perm_tensor, mobi_tensor, \
            func_tensor, *h_tensor, *rhograv_tensor, nz, nx);

  }
};
REGISTER_KERNEL_BUILDER(Name("UpwlapOp").Device(DEVICE_CPU), UpwlapOpOp);


REGISTER_OP("UpwlapOpGrad")

.Input("grad_out : double")
.Input("out : double")
.Input("perm : double")
.Input("mobi : double")
.Input("func : double")
.Input("h : double")
.Input("rhograv : double")
.Output("grad_perm : double")
.Output("grad_mobi : double")
.Output("grad_func : double")
.Output("grad_h : double")
.Output("grad_rhograv : double");
class UpwlapOpGradOp : public OpKernel {
 private:

 public:
  explicit UpwlapOpGradOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {


    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& perm = context->input(2);
    const Tensor& mobi = context->input(3);
    const Tensor& func = context->input(4);
    const Tensor& h = context->input(5);
    const Tensor& rhograv = context->input(6);


    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& perm_shape = perm.shape();
    const TensorShape& mobi_shape = mobi.shape();
    const TensorShape& func_shape = func.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& rhograv_shape = rhograv.shape();


    DCHECK_EQ(grad_out_shape.dims(), 2);
    DCHECK_EQ(out_shape.dims(), 2);
    DCHECK_EQ(perm_shape.dims(), 2);
    DCHECK_EQ(mobi_shape.dims(), 2);
    DCHECK_EQ(func_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(rhograv_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);

    // create output shape
    int nz = func_shape.dim_size(0), nx = func_shape.dim_size(1);

    TensorShape grad_perm_shape(perm_shape);
    TensorShape grad_mobi_shape(mobi_shape);
    TensorShape grad_func_shape(func_shape);
    TensorShape grad_h_shape(h_shape);
    TensorShape grad_rhograv_shape(rhograv_shape);

    // create output tensor

    Tensor* grad_perm = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_perm_shape, &grad_perm));
    Tensor* grad_mobi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_mobi_shape, &grad_mobi));
    Tensor* grad_func = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_func_shape, &grad_func));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    Tensor* grad_rhograv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_rhograv_shape, &grad_rhograv));

    // get the corresponding Eigen tensors for data access

    auto perm_tensor = perm.flat<double>().data();
    auto mobi_tensor = mobi.flat<double>().data();
    auto func_tensor = func.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto rhograv_tensor = rhograv.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_perm_tensor = grad_perm->flat<double>().data();
    auto grad_mobi_tensor = grad_mobi->flat<double>().data();
    auto grad_func_tensor = grad_func->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();
    auto grad_rhograv_tensor = grad_rhograv->flat<double>().data();

    // implement your backward function here

    // TODO:
    backward(grad_out_tensor, perm_tensor, mobi_tensor, \
             func_tensor, *h_tensor, *rhograv_tensor, grad_perm_tensor, grad_mobi_tensor, \
             grad_func_tensor, nz, nx);

  }
};
REGISTER_KERNEL_BUILDER(Name("UpwlapOpGrad").Device(DEVICE_CPU), UpwlapOpGradOp);

