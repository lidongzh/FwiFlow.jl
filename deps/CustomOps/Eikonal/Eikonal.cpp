#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>


#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif
using namespace tensorflow;
#include "Eikonal.h"


REGISTER_OP("Eikonal")

.Input("f : double")
.Input("srcx : int64")
.Input("srcy : int64")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("u : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle f_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &f_shape));
        shape_inference::ShapeHandle srcx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &srcx_shape));
        shape_inference::ShapeHandle srcy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &srcy_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("EikonalGrad")

.Input("grad_u : double")
.Input("u : double")
.Input("f : double")
.Input("srcx : int64")
.Input("srcy : int64")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("grad_f : double")
.Output("grad_srcx : int64")
.Output("grad_srcy : int64")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_h : double");


class EikonalOp : public OpKernel {
private:
  
public:
  explicit EikonalOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& f = context->input(0);
    const Tensor& srcx = context->input(1);
    const Tensor& srcy = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& f_shape = f.shape();
    const TensorShape& srcx_shape = srcx.shape();
    const TensorShape& srcy_shape = srcy.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(srcx_shape.dims(), 0);
    DCHECK_EQ(srcy_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape u_shape({f_shape.dim_size(0)});
            
    // create output tensor
    
    Tensor* u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &u));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto srcx_tensor = srcx.flat<int64>().data();
    auto srcy_tensor = srcy.flat<int64>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto u_tensor = u->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(u_tensor, f_tensor, *m_tensor, *n_tensor, *h_tensor, *srcx_tensor - 1, *srcy_tensor - 1);

  }
};
REGISTER_KERNEL_BUILDER(Name("Eikonal").Device(DEVICE_CPU), EikonalOp);



class EikonalGradOp : public OpKernel {
private:
  
public:
  explicit EikonalGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_u = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& f = context->input(2);
    const Tensor& srcx = context->input(3);
    const Tensor& srcy = context->input(4);
    const Tensor& m = context->input(5);
    const Tensor& n = context->input(6);
    const Tensor& h = context->input(7);
    
    
    const TensorShape& grad_u_shape = grad_u.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& srcx_shape = srcx.shape();
    const TensorShape& srcy_shape = srcy.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_u_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(srcx_shape.dims(), 0);
    DCHECK_EQ(srcy_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_f_shape(f_shape);
    TensorShape grad_srcx_shape(srcx_shape);
    TensorShape grad_srcy_shape(srcy_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_f_shape, &grad_f));
    Tensor* grad_srcx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_srcx_shape, &grad_srcx));
    Tensor* grad_srcy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_srcy_shape, &grad_srcy));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto srcx_tensor = srcx.flat<int64>().data();
    auto srcy_tensor = srcy.flat<int64>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_u_tensor = grad_u.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:

    backward(
      grad_f_tensor, grad_u_tensor,
      u_tensor, f_tensor, *m_tensor, *n_tensor, *h_tensor, *srcx_tensor - 1, *srcy_tensor - 1);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("EikonalGrad").Device(DEVICE_CPU), EikonalGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class EikonalOpGPU : public OpKernel {
private:
  
public:
  explicit EikonalOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& f = context->input(0);
    const Tensor& srcx = context->input(1);
    const Tensor& srcy = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& f_shape = f.shape();
    const TensorShape& srcx_shape = srcx.shape();
    const TensorShape& srcy_shape = srcy.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(srcx_shape.dims(), 0);
    DCHECK_EQ(srcy_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape u_shape({-1});
            
    // create output tensor
    
    Tensor* u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &u));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto srcx_tensor = srcx.flat<int64>().data();
    auto srcy_tensor = srcy.flat<int64>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto u_tensor = u->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("Eikonal").Device(DEVICE_GPU), EikonalOpGPU);

class EikonalGradOpGPU : public OpKernel {
private:
  
public:
  explicit EikonalGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_u = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& f = context->input(2);
    const Tensor& srcx = context->input(3);
    const Tensor& srcy = context->input(4);
    const Tensor& m = context->input(5);
    const Tensor& n = context->input(6);
    const Tensor& h = context->input(7);
    
    
    const TensorShape& grad_u_shape = grad_u.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& srcx_shape = srcx.shape();
    const TensorShape& srcy_shape = srcy.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_u_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(srcx_shape.dims(), 0);
    DCHECK_EQ(srcy_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_f_shape(f_shape);
    TensorShape grad_srcx_shape(srcx_shape);
    TensorShape grad_srcy_shape(srcy_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_f_shape, &grad_f));
    Tensor* grad_srcx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_srcx_shape, &grad_srcx));
    Tensor* grad_srcy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_srcy_shape, &grad_srcy));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto srcx_tensor = srcx.flat<int64>().data();
    auto srcy_tensor = srcy.flat<int64>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_u_tensor = grad_u.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("EikonalGrad").Device(DEVICE_GPU), EikonalGradOpGPU);

#endif