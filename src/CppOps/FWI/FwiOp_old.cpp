#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
using namespace std;
using std::string;
using namespace tensorflow;
// #include "ADEL.h"
#include "FwiOp.h"

REGISTER_OP("FwiOp")

    .Input("cp : double")
    .Input("cs : double")
    .Input("den : double")
    .Input("stf: double")
    .Input("prjdir: string")
    .Output("res : double")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle cp_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &cp_shape));
      shape_inference::ShapeHandle cs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &cs_shape));
      shape_inference::ShapeHandle den_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &den_shape));
      shape_inference::ShapeHandle src_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &src_shape));

      c->set_output(0, c->Scalar());
      return Status::OK();
    });
class FwiOpOp : public OpKernel {
 private:
 public:
  explicit FwiOpOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DCHECK_EQ(5, context->num_inputs());

    const Tensor& cp = context->input(0);
    const Tensor& cs = context->input(1);
    const Tensor& den = context->input(2);
    const Tensor& src = context->input(3);
    const Tensor& prjdir = context->input(4);

    const TensorShape& cp_shape = cp.shape();
    const TensorShape& cs_shape = cs.shape();
    const TensorShape& den_shape = den.shape();
    const TensorShape& src_shape = src.shape();

    DCHECK_EQ(cp_shape.dims(), 2);
    DCHECK_EQ(cs_shape.dims(), 2);
    DCHECK_EQ(den_shape.dims(), 2);
    DCHECK_EQ(src_shape.dims(), 1);

    // extra check

    // create output shape

    TensorShape res_shape({});

    // create output tensor

    Tensor* res = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, res_shape, &res));

    // get the corresponding Eigen tensors for data access

    auto cp_tensor = cp.flat<double>().data();
    auto cs_tensor = cs.flat<double>().data();
    auto den_tensor = den.flat<double>().data();
    auto src_tensor = src.flat<double>().data();
    auto prjdir_tensor = prjdir.flat<string>().data();
    auto res_tensor = res->flat<double>().data();

    // implement your forward function here

    // TODO:
    string dir(getenv("PARAMDIR"));
    std::cout << dir << std::endl;

    std::cout << *prjdir_tensor << " !!!!!!!!" << std::endl;
    forward(res_tensor, cp_tensor, cs_tensor, den_tensor, dir);
  }
};
REGISTER_KERNEL_BUILDER(Name("FwiOp").Device(DEVICE_CPU), FwiOpOp);

REGISTER_OP("FwiOpGrad")

    .Input("grad_res : double")
    .Input("res : double")
    .Input("cp : double")
    .Input("cs : double")
    .Input("den : double")
    .Input("src: double")
    .Input("prjdir: string")
    .Output("grad_cp : double")
    .Output("grad_cs : double")
    .Output("grad_den : double")
    .Output("grad_src: double")
    .Output("grad_prjdir: string");
class FwiOpGradOp : public OpKernel {
 private:
 public:
  explicit FwiOpGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_res = context->input(0);
    const Tensor& res = context->input(1);
    const Tensor& cp = context->input(2);
    const Tensor& cs = context->input(3);
    const Tensor& den = context->input(4);
    const Tensor& src = context->input(5);
    const Tensor& prjdir = context->input(6);

    const TensorShape& grad_res_shape = grad_res.shape();
    const TensorShape& res_shape = res.shape();
    const TensorShape& cp_shape = cp.shape();
    const TensorShape& cs_shape = cs.shape();
    const TensorShape& den_shape = den.shape();
    const TensorShape& src_shape = src.shape();
    const TensorShape& prjdir_shape = prjdir.shape();

    DCHECK_EQ(grad_res_shape.dims(), 0);
    DCHECK_EQ(res_shape.dims(), 0);
    DCHECK_EQ(cp_shape.dims(), 2);
    DCHECK_EQ(cs_shape.dims(), 2);
    DCHECK_EQ(den_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);

    // create output shape

    TensorShape grad_cp_shape(cp_shape);
    TensorShape grad_cs_shape(cs_shape);
    TensorShape grad_den_shape(den_shape);
    TensorShape grad_src_shape(den_shape);
    TensorShape grad_prjdir_shape(den_shape);

    // create output tensor

    Tensor* grad_cp = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, grad_cp_shape, &grad_cp));
    Tensor* grad_cs = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, grad_cs_shape, &grad_cs));
    Tensor* grad_den = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, grad_den_shape, &grad_den));
    Tensor* grad_src = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, grad_src_shape, &grad_src));
    Tensor* grad_prjdir = NULL;
    OP_REQUIRES_OK(
        context, context->allocate_output(4, grad_prjdir_shape, &grad_prjdir));

    // get the corresponding Eigen tensors for data access

    auto cp_tensor = cp.flat<double>().data();
    auto cs_tensor = cs.flat<double>().data();
    auto den_tensor = den.flat<double>().data();
    auto src_tensor = src.flat<double>().data();
    auto prjdir_tensor = prjdir.flat<string>().data();
    auto grad_res_tensor = grad_res.flat<double>().data();
    auto res_tensor = res.flat<double>().data();
    auto grad_cp_tensor = grad_cp->flat<double>().data();
    auto grad_cs_tensor = grad_cs->flat<double>().data();
    auto grad_den_tensor = grad_den->flat<double>().data();
    auto grad_src_tensor = grad_src->flat<double>().data();
    auto grad_prjdir_tensor = grad_prjdir->flat<string>().data();

    // implement your backward function here

    // TODO:
    // void backward(float *d_Cp, const float *Cp, const float *Cs, const float
    // *Den, string dir){
    string dir(getenv("PARAMDIR"));
    std::cout << dir << std::endl;
    backward(grad_cp_tensor, cp_tensor, cs_tensor, den_tensor, dir);
  }
};
REGISTER_KERNEL_BUILDER(Name("FwiOpGrad").Device(DEVICE_CPU), FwiOpGradOp);
