#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
using namespace tensorflow;
// #include "ADEL.h"
#include "PoissonOp.h"

REGISTER_OP("PoissonOp")

.Input("coef : double")
.Input("g : double")
.Input("h : double")
.Input("rhograv : double")
.Input("index : int64")
.Output("p : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {

    shape_inference::ShapeHandle coef_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &coef_shape));
    shape_inference::ShapeHandle g_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &g_shape));
    shape_inference::ShapeHandle h_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &h_shape));
    shape_inference::ShapeHandle rhograv_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));
    shape_inference::ShapeHandle index_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &h_shape));

    c->set_output(0, c->Matrix(-1, -1));
    // c->set_output(0, c->Matrix(c->Dim(coef_shape, 0), c->Dim(coef_shape, 1)));
    return Status::OK();
});
class PoissonOpOp : public OpKernel {
  private:

  public:
    explicit PoissonOpOp(OpKernelConstruction *context) : OpKernel(context) {

    }

    void Compute(OpKernelContext *context) override {
        DCHECK_EQ(5, context->num_inputs());


        const Tensor &coef = context->input(0);
        const Tensor &g = context->input(1);
        const Tensor &h = context->input(2);
        const Tensor &rhograv = context->input(3);
        const Tensor &index = context->input(4);


        const TensorShape &coef_shape = coef.shape();
        const TensorShape &g_shape = g.shape();
        const TensorShape &h_shape = h.shape();
        const TensorShape &rhograv_shape = rhograv.shape();
        const TensorShape &index_shape = index.shape();


        DCHECK_EQ(coef_shape.dims(), 2);
        DCHECK_EQ(g_shape.dims(), 2);
        DCHECK_EQ(h_shape.dims(), 0);
        DCHECK_EQ(rhograv_shape.dims(), 0);
        DCHECK_EQ(index_shape.dims(), 0);

        // extra check

        // create output shape
        int nz = coef_shape.dim_size(0), nx = coef_shape.dim_size(1);

        // TensorShape p_shape({ -1, -1});

        // create output tensor
        TensorShape p_shape({nz, nx});


        Tensor *p = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, p_shape, &p));

        // get the corresponding Eigen tensors for data access

        auto coef_tensor = coef.flat<double>().data();
        auto g_tensor = g.flat<double>().data();
        auto h_tensor = h.flat<double>().data();
        auto rhograv_tensor = rhograv.flat<double>().data();
        auto index_tensor = index.flat<int64>().data();
        auto p_tensor = p->flat<double>().data();


        // implement your forward function here

        // TODO:
        forward(p_tensor, coef_tensor, g_tensor, *h_tensor, *rhograv_tensor, *index_tensor, nz, nx);

    }
};
REGISTER_KERNEL_BUILDER(Name("PoissonOp").Device(DEVICE_CPU), PoissonOpOp);


REGISTER_OP("PoissonOpGrad")

.Input("grad_p : double")
.Input("p : double")
.Input("coef : double")
.Input("g : double")
.Input("h : double")
.Input("rhograv : double")
.Input("index : int64")
.Output("grad_coef : double")
.Output("grad_g : double")
.Output("grad_h : double")
.Output("grad_rhograv : double")
.Output("grad_index : int64");
class PoissonOpGradOp : public OpKernel {
  private:

  public:
    explicit PoissonOpGradOp(OpKernelConstruction *context) : OpKernel(context) {

    }

    void Compute(OpKernelContext *context) override {


        const Tensor &grad_p = context->input(0);
        const Tensor &p = context->input(1);
        const Tensor &coef = context->input(2);
        const Tensor &g = context->input(3);
        const Tensor &h = context->input(4);
        const Tensor &rhograv = context->input(5);
        const Tensor &index = context->input(6);


        const TensorShape &grad_p_shape = grad_p.shape();
        const TensorShape &p_shape = p.shape();
        const TensorShape &coef_shape = coef.shape();
        const TensorShape &g_shape = g.shape();
        const TensorShape &h_shape = h.shape();
        const TensorShape &rhograv_shape = rhograv.shape();
        const TensorShape &index_shape = index.shape();


        DCHECK_EQ(grad_p_shape.dims(), 2);
        DCHECK_EQ(p_shape.dims(), 2);
        DCHECK_EQ(coef_shape.dims(), 2);
        DCHECK_EQ(g_shape.dims(), 2);
        DCHECK_EQ(h_shape.dims(), 0);
        DCHECK_EQ(rhograv_shape.dims(), 0);
        DCHECK_EQ(index_shape.dims(), 0);

        // extra check
        // int m = Example.dim_size(0);

        // create output shape
        int nz = coef_shape.dim_size(0), nx = coef_shape.dim_size(1);

        TensorShape grad_coef_shape(coef_shape);
        TensorShape grad_g_shape(g_shape);
        TensorShape grad_h_shape(h_shape);
        TensorShape grad_rhograv_shape(rhograv_shape);
        TensorShape grad_index_shape(index_shape);

        // create output tensor

        Tensor *grad_coef = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, grad_coef_shape, &grad_coef));
        Tensor *grad_g = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, grad_g_shape, &grad_g));
        Tensor *grad_h = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, grad_h_shape, &grad_h));
        Tensor *grad_rhograv = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(3, grad_rhograv_shape, &grad_rhograv));
        Tensor *grad_index = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(4, grad_index_shape, &grad_index));

        // get the corresponding Eigen tensors for data access

        auto coef_tensor = coef.flat<double>().data();
        auto g_tensor = g.flat<double>().data();
        auto h_tensor = h.flat<double>().data();
        auto rhograv_tensor = rhograv.flat<double>().data();
        auto index_tensor = index.flat<int64>().data();
        auto grad_p_tensor = grad_p.flat<double>().data();
        auto p_tensor = p.flat<double>().data();
        auto grad_coef_tensor = grad_coef->flat<double>().data();
        auto grad_g_tensor = grad_g->flat<double>().data();
        auto grad_h_tensor = grad_h->flat<double>().data();
        auto grad_rhograv_tensor = grad_rhograv->flat<double>().data();
        auto grad_index_tensor = grad_index->flat<int64>().data();

        // implement your backward function here

        // TODO:
        backward(grad_p_tensor, p_tensor, coef_tensor, g_tensor, *h_tensor, *rhograv_tensor, *index_tensor, nz, nx,
                 grad_coef_tensor, grad_g_tensor);

    }
};
REGISTER_KERNEL_BUILDER(Name("PoissonOpGrad").Device(DEVICE_CPU), PoissonOpGradOp);

