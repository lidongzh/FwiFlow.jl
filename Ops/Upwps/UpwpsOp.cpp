#include <cmath>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
using namespace tensorflow;
// #include "ADEL.h"
#include "UpwpsOp.h"

REGISTER_OP("UpwpsOp")

    .Input("permi : double")
    .Input("mobi : double")
    .Input("src : double")
    .Input("funcref : double")
    .Input("h : double")
    .Input("rhograv : double")
    .Input("index : int64")
    .Output("pres : double")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      shape_inference::ShapeHandle permi_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &permi_shape));
      shape_inference::ShapeHandle mobi_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &mobi_shape));
      shape_inference::ShapeHandle src_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &src_shape));
      shape_inference::ShapeHandle funcref_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &funcref_shape));
      shape_inference::ShapeHandle h_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &h_shape));
      shape_inference::ShapeHandle rhograv_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &rhograv_shape));
      shape_inference::ShapeHandle index_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &index_shape));

      c->set_output(0, c->Matrix(-1, -1));
      return Status::OK();
    });
class UpwpsOpOp : public OpKernel {
 private:
 public:
  explicit UpwpsOpOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    DCHECK_EQ(7, context->num_inputs());

    const Tensor &permi = context->input(0);
    const Tensor &mobi = context->input(1);
    const Tensor &src = context->input(2);
    const Tensor &funcref = context->input(3);
    const Tensor &h = context->input(4);
    const Tensor &rhograv = context->input(5);
    const Tensor &index = context->input(6);

    const TensorShape &permi_shape = permi.shape();
    const TensorShape &mobi_shape = mobi.shape();
    const TensorShape &src_shape = src.shape();
    const TensorShape &funcref_shape = funcref.shape();
    const TensorShape &h_shape = h.shape();
    const TensorShape &rhograv_shape = rhograv.shape();
    const TensorShape &index_shape = index.shape();

    DCHECK_EQ(permi_shape.dims(), 2);
    DCHECK_EQ(mobi_shape.dims(), 2);
    DCHECK_EQ(src_shape.dims(), 2);
    DCHECK_EQ(funcref_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(rhograv_shape.dims(), 0);
    DCHECK_EQ(index_shape.dims(), 0);

    // extra check

    // create output shape
    int nz = permi_shape.dim_size(0), nx = permi_shape.dim_size(1);

    TensorShape pres_shape({nz, nx});

    // create output tensor

    Tensor *pres = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, pres_shape, &pres));

    // get the corresponding Eigen tensors for data access

    auto permi_tensor = permi.flat<double>().data();
    auto mobi_tensor = mobi.flat<double>().data();
    auto src_tensor = src.flat<double>().data();
    auto funcref_tensor = funcref.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto rhograv_tensor = rhograv.flat<double>().data();
    auto index_tensor = index.flat<int64>().data();
    auto pres_tensor = pres->flat<double>().data();

    // implement your forward function here

    // TODO:
    forward(pres_tensor, permi_tensor, mobi_tensor, src_tensor, funcref_tensor,
            *h_tensor, *rhograv_tensor, *index_tensor, nz, nx);
  }
};
REGISTER_KERNEL_BUILDER(Name("UpwpsOp").Device(DEVICE_CPU), UpwpsOpOp);

REGISTER_OP("UpwpsOpGrad")

    .Input("grad_pres : double")
    .Input("pres : double")
    .Input("permi : double")
    .Input("mobi : double")
    .Input("src : double")
    .Input("funcref : double")
    .Input("h : double")
    .Input("rhograv : double")
    .Input("index : int64")
    .Output("grad_permi : double")
    .Output("grad_mobi : double")
    .Output("grad_src : double")
    .Output("grad_funcref : double")
    .Output("grad_h : double")
    .Output("grad_rhograv : double")
    .Output("grad_index : int64");
class UpwpsOpGradOp : public OpKernel {
 private:
 public:
  explicit UpwpsOpGradOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    const Tensor &grad_pres = context->input(0);
    const Tensor &pres = context->input(1);
    const Tensor &permi = context->input(2);
    const Tensor &mobi = context->input(3);
    const Tensor &src = context->input(4);
    const Tensor &funcref = context->input(5);
    const Tensor &h = context->input(6);
    const Tensor &rhograv = context->input(7);
    const Tensor &index = context->input(8);

    const TensorShape &grad_pres_shape = grad_pres.shape();
    const TensorShape &pres_shape = pres.shape();
    const TensorShape &permi_shape = permi.shape();
    const TensorShape &mobi_shape = mobi.shape();
    const TensorShape &src_shape = src.shape();
    const TensorShape &funcref_shape = funcref.shape();
    const TensorShape &h_shape = h.shape();
    const TensorShape &rhograv_shape = rhograv.shape();
    const TensorShape &index_shape = index.shape();

    DCHECK_EQ(grad_pres_shape.dims(), 2);
    DCHECK_EQ(pres_shape.dims(), 2);
    DCHECK_EQ(permi_shape.dims(), 2);
    DCHECK_EQ(mobi_shape.dims(), 2);
    DCHECK_EQ(src_shape.dims(), 2);
    DCHECK_EQ(funcref_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(rhograv_shape.dims(), 0);
    DCHECK_EQ(index_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);

    // create output shape
    int nz = permi_shape.dim_size(0), nx = permi_shape.dim_size(1);

    TensorShape grad_permi_shape(permi_shape);
    TensorShape grad_mobi_shape(mobi_shape);
    TensorShape grad_src_shape(src_shape);
    TensorShape grad_funcref_shape(funcref_shape);
    TensorShape grad_h_shape(h_shape);
    TensorShape grad_rhograv_shape(rhograv_shape);
    TensorShape grad_index_shape(index_shape);

    // create output tensor

    Tensor *grad_permi = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, grad_permi_shape, &grad_permi));
    Tensor *grad_mobi = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, grad_mobi_shape, &grad_mobi));
    Tensor *grad_src = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, grad_src_shape, &grad_src));
    Tensor *grad_funcref = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_funcref_shape,
                                                     &grad_funcref));
    Tensor *grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_h_shape, &grad_h));
    Tensor *grad_rhograv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_rhograv_shape,
                                                     &grad_rhograv));
    Tensor *grad_index = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(6, grad_index_shape, &grad_index));

    // get the corresponding Eigen tensors for data access

    auto permi_tensor = permi.flat<double>().data();
    auto mobi_tensor = mobi.flat<double>().data();
    auto src_tensor = src.flat<double>().data();
    auto funcref_tensor = funcref.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto rhograv_tensor = rhograv.flat<double>().data();
    auto index_tensor = index.flat<int64>().data();
    auto grad_pres_tensor = grad_pres.flat<double>().data();
    auto pres_tensor = pres.flat<double>().data();
    auto grad_permi_tensor = grad_permi->flat<double>().data();
    auto grad_mobi_tensor = grad_mobi->flat<double>().data();
    auto grad_src_tensor = grad_src->flat<double>().data();
    auto grad_funcref_tensor = grad_funcref->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();
    auto grad_rhograv_tensor = grad_rhograv->flat<double>().data();
    auto grad_index_tensor = grad_index->flat<int64>().data();

    // implement your backward function here

    // TODO:
    backward(grad_pres_tensor, pres_tensor, permi_tensor, mobi_tensor,
             src_tensor, funcref_tensor, *h_tensor, *rhograv_tensor,
             *index_tensor, nz, nx, grad_permi_tensor, grad_mobi_tensor,
             grad_src_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("UpwpsOpGrad").Device(DEVICE_CPU), UpwpsOpGradOp);
