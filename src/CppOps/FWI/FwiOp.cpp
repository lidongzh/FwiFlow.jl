#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
// using namespace std;
using std::string;
using namespace tensorflow;
// #include "ADEL.h"
#include "FwiOp.h"

REGISTER_OP("FwiOp")

    .Input("lambda : double")
    .Input("mu : double")
    .Input("den : double")
    .Input("stf : double")
    .Input("gpu_id : int32")
    .Input("shot_ids : int32")
    .Input("para_fname : string")
    .Output("misfit : double")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle lambda_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &lambda_shape));
      shape_inference::ShapeHandle mu_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &mu_shape));
      shape_inference::ShapeHandle den_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &den_shape));
      shape_inference::ShapeHandle stf_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &stf_shape));
      shape_inference::ShapeHandle gpu_id_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &gpu_id_shape));
      shape_inference::ShapeHandle shot_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &shot_ids_shape));

      c->set_output(0, c->Scalar());
      return Status::OK();
    });
class FwiOpOp : public OpKernel {
 private:
 public:
  explicit FwiOpOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DCHECK_EQ(7, context->num_inputs());

    const Tensor& lambda = context->input(0);
    const Tensor& mu = context->input(1);
    const Tensor& den = context->input(2);
    const Tensor& stf = context->input(3);
    const Tensor& gpu_id = context->input(4);
    const Tensor& shot_ids = context->input(5);
    const Tensor& para_fname = context->input(6);

    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& den_shape = den.shape();
    const TensorShape& stf_shape = stf.shape();
    const TensorShape& gpu_id_shape = gpu_id.shape();
    const TensorShape& shot_ids_shape = shot_ids.shape();

    DCHECK_EQ(lambda_shape.dims(), 2);
    DCHECK_EQ(mu_shape.dims(), 2);
    DCHECK_EQ(den_shape.dims(), 2);
    DCHECK_EQ(stf_shape.dims(), 2);
    DCHECK_EQ(gpu_id_shape.dims(), 0);
    DCHECK_EQ(shot_ids_shape.dims(), 1);

    // extra check

    // create output shape
    // int nz = lambda_shape.dim_size(0), nx = lambda_shape.dim_size(1);
    int group_size = shot_ids_shape.dim_size(0);

    TensorShape misfit_shape({});

    // create output tensor

    Tensor* misfit = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, misfit_shape, &misfit));

    // get the corresponding Eigen tensors for data access

    auto lambda_tensor = lambda.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto den_tensor = den.flat<double>().data();
    auto stf_tensor = stf.flat<double>().data();
    auto gpu_id_tensor = gpu_id.flat<int32>().data();
    auto shot_ids_tensor = shot_ids.flat<int32>().data();
    auto para_fname_tensor = para_fname.flat<string>().data();
    auto misfit_tensor = misfit->flat<double>().data();

    // implement your forward function here

    // TODO:
    // std::cout << *para_fname_tensor << " !!!!!!!!" << std::endl;
    forward(misfit_tensor, lambda_tensor, mu_tensor, den_tensor, stf_tensor,
            *gpu_id_tensor, group_size, shot_ids_tensor,
            string(*para_fname_tensor));
  }
};
REGISTER_KERNEL_BUILDER(Name("FwiOp").Device(DEVICE_CPU), FwiOpOp);

REGISTER_OP("FwiOpGrad")

    .Input("grad_misfit : double")
    .Input("misfit : double")
    .Input("lambda : double")
    .Input("mu : double")
    .Input("den : double")
    .Input("stf : double")
    .Input("gpu_id : int32")
    .Input("shot_ids : int32")
    .Input("para_fname: string")
    .Output("grad_lambda : double")
    .Output("grad_mu : double")
    .Output("grad_den : double")
    .Output("grad_stf : double")
    .Output("grad_gpu_id : int32")
    .Output("grad_shot_ids : int32")
    .Output("grad_para_fname:string");
class FwiOpGradOp : public OpKernel {
 private:
 public:
  explicit FwiOpGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_misfit = context->input(0);
    const Tensor& misfit = context->input(1);
    const Tensor& lambda = context->input(2);
    const Tensor& mu = context->input(3);
    const Tensor& den = context->input(4);
    const Tensor& stf = context->input(5);
    const Tensor& gpu_id = context->input(6);
    const Tensor& shot_ids = context->input(7);
    const Tensor& para_fname = context->input(8);

    const TensorShape& grad_misfit_shape = grad_misfit.shape();
    const TensorShape& misfit_shape = misfit.shape();
    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& den_shape = den.shape();
    const TensorShape& stf_shape = stf.shape();
    const TensorShape& gpu_id_shape = gpu_id.shape();
    const TensorShape& shot_ids_shape = shot_ids.shape();
    const TensorShape& para_fname_shape = para_fname.shape();

    DCHECK_EQ(grad_misfit_shape.dims(), 0);
    DCHECK_EQ(misfit_shape.dims(), 0);
    DCHECK_EQ(lambda_shape.dims(), 2);
    DCHECK_EQ(mu_shape.dims(), 2);
    DCHECK_EQ(den_shape.dims(), 2);
    DCHECK_EQ(stf_shape.dims(), 2);
    DCHECK_EQ(gpu_id_shape.dims(), 0);
    DCHECK_EQ(shot_ids_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);

    // create output shape
    int group_size = shot_ids_shape.dim_size(0);

    TensorShape grad_lambda_shape(lambda_shape);
    TensorShape grad_mu_shape(mu_shape);
    TensorShape grad_den_shape(den_shape);
    TensorShape grad_stf_shape(stf_shape);
    TensorShape grad_gpu_id_shape(gpu_id_shape);
    TensorShape grad_shot_ids_shape(shot_ids_shape);
    TensorShape grad_para_fname_shape(para_fname_shape);

    // create output tensor

    Tensor* grad_lambda = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, grad_lambda_shape, &grad_lambda));
    Tensor* grad_mu = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, grad_mu_shape, &grad_mu));
    Tensor* grad_den = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, grad_den_shape, &grad_den));
    Tensor* grad_stf = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, grad_stf_shape, &grad_stf));
    Tensor* grad_gpu_id = NULL;
    OP_REQUIRES_OK(
        context, context->allocate_output(4, grad_gpu_id_shape, &grad_gpu_id));
    Tensor* grad_shot_ids = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_shot_ids_shape,
                                                     &grad_shot_ids));
    Tensor* grad_para_fname = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_para_fname_shape,
                                                     &grad_para_fname));

    // get the corresponding Eigen tensors for data access

    auto lambda_tensor = lambda.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto den_tensor = den.flat<double>().data();
    auto stf_tensor = stf.flat<double>().data();
    auto gpu_id_tensor = gpu_id.flat<int32>().data();
    auto shot_ids_tensor = shot_ids.flat<int32>().data();
    auto para_fname_tensor = para_fname.flat<string>().data();
    auto grad_misfit_tensor = grad_misfit.flat<double>().data();
    auto misfit_tensor = misfit.flat<double>().data();
    auto grad_lambda_tensor = grad_lambda->flat<double>().data();
    auto grad_mu_tensor = grad_mu->flat<double>().data();
    auto grad_den_tensor = grad_den->flat<double>().data();
    auto grad_stf_tensor = grad_stf->flat<double>().data();
    auto grad_gpu_id_tensor = grad_gpu_id->flat<int32>().data();
    auto grad_shot_ids_tensor = grad_shot_ids->flat<int32>().data();
    auto grad_para_fname_tensor = grad_para_fname->flat<string>().data();

    // implement your backward function here
    // std::cout << *grad_para_fname_tensor << " !!!!!" << std::endl;
    // TODO:
    backward(grad_lambda_tensor, grad_mu_tensor, grad_den_tensor, grad_stf_tensor,
             lambda_tensor, mu_tensor, den_tensor, stf_tensor, *gpu_id_tensor,
             group_size, shot_ids_tensor, string(*para_fname_tensor));
  }
};
REGISTER_KERNEL_BUILDER(Name("FwiOpGrad").Device(DEVICE_CPU), FwiOpGradOp);

// To generate observed data for synthetic tests
REGISTER_OP("FwiObsOp")

    .Input("lambda : double")
    .Input("mu : double")
    .Input("den : double")
    .Input("stf : double")
    .Input("gpu_id : int32")
    .Input("shot_ids : int32")
    .Input("para_fname : string")
    .Output("misfit : double")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle lambda_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &lambda_shape));
      shape_inference::ShapeHandle mu_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &mu_shape));
      shape_inference::ShapeHandle den_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &den_shape));
      shape_inference::ShapeHandle stf_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &stf_shape));
      shape_inference::ShapeHandle gpu_id_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &gpu_id_shape));
      shape_inference::ShapeHandle shot_ids_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &shot_ids_shape));

      c->set_output(0, c->Scalar());
      return Status::OK();
    });
class FwiObsOpOp : public OpKernel {
 private:
 public:
  explicit FwiObsOpOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DCHECK_EQ(7, context->num_inputs());

    const Tensor& lambda = context->input(0);
    const Tensor& mu = context->input(1);
    const Tensor& den = context->input(2);
    const Tensor& stf = context->input(3);
    const Tensor& gpu_id = context->input(4);
    const Tensor& shot_ids = context->input(5);
    const Tensor& para_fname = context->input(6);

    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& den_shape = den.shape();
    const TensorShape& stf_shape = stf.shape();
    const TensorShape& gpu_id_shape = gpu_id.shape();
    const TensorShape& shot_ids_shape = shot_ids.shape();

    DCHECK_EQ(lambda_shape.dims(), 2);
    DCHECK_EQ(mu_shape.dims(), 2);
    DCHECK_EQ(den_shape.dims(), 2);
    DCHECK_EQ(stf_shape.dims(), 2);
    DCHECK_EQ(gpu_id_shape.dims(), 0);
    DCHECK_EQ(shot_ids_shape.dims(), 1);

    // extra check

    // create output shape
    // int nz = lambda_shape.dim_size(0), nx = lambda_shape.dim_size(1);
    int group_size = shot_ids_shape.dim_size(0);

    TensorShape misfit_shape({1});

    // create output tensor

    Tensor* misfit = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, misfit_shape, &misfit));

    // get the corresponding Eigen tensors for data access

    auto lambda_tensor = lambda.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto den_tensor = den.flat<double>().data();
    auto stf_tensor = stf.flat<double>().data();
    auto gpu_id_tensor = gpu_id.flat<int32>().data();
    auto shot_ids_tensor = shot_ids.flat<int32>().data();
    auto para_fname_tensor = para_fname.flat<string>().data();
    auto misfit_tensor = misfit->flat<double>().data();

    // implement your forward function here

    // TODO:
    // std::cout << *para_fname_tensor << " !!!!!!!!" << std::endl;
    obscalc(misfit_tensor, lambda_tensor, mu_tensor, den_tensor, stf_tensor,
            *gpu_id_tensor, group_size, shot_ids_tensor,
            string(*para_fname_tensor));
    *misfit_tensor = 0.0;
  }
};
REGISTER_KERNEL_BUILDER(Name("FwiObsOp").Device(DEVICE_CPU), FwiObsOpOp);
