#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "SaturationNn.h"


REGISTER_OP("SaturationNn")
.Input("s0 : double")
.Input("dporodt : double")
.Input("pt : double")
.Input("perm : double")
.Input("poro : double")
.Input("qw : double")
.Input("qo : double")
.Input("muw : double")
.Input("muo : double")
.Input("sref : double")
.Input("thetaw : double")
.Input("configw : int64")
.Input("thetao : double")
.Input("configo : int64")
.Input("dt : double")
.Input("h : double")
.Output("sat : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle s0_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &s0_shape));
        shape_inference::ShapeHandle dporodt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &dporodt_shape));
        shape_inference::ShapeHandle pt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &pt_shape));
        shape_inference::ShapeHandle perm_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &perm_shape));
        shape_inference::ShapeHandle poro_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &poro_shape));
        shape_inference::ShapeHandle qw_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &qw_shape));
        shape_inference::ShapeHandle qo_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &qo_shape));
        shape_inference::ShapeHandle muw_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &muw_shape));
        shape_inference::ShapeHandle muo_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &muo_shape));
        shape_inference::ShapeHandle sref_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 2, &sref_shape));
        shape_inference::ShapeHandle thetaw_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 1, &thetaw_shape));
        shape_inference::ShapeHandle configw_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(11), 1, &configw_shape));
        shape_inference::ShapeHandle thetao_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(12), 1, &thetao_shape));
        shape_inference::ShapeHandle configo_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(13), 1, &configo_shape));
        shape_inference::ShapeHandle dt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(14), 0, &dt_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(15), 0, &h_shape));

        c->set_output(0, c->Matrix(-1,-1));
    return Status::OK();
  });

REGISTER_OP("SaturationNnGrad")
.Input("grad_sat : double")
.Input("sat : double")
.Input("s0 : double")
.Input("dporodt : double")
.Input("pt : double")
.Input("perm : double")
.Input("poro : double")
.Input("qw : double")
.Input("qo : double")
.Input("muw : double")
.Input("muo : double")
.Input("sref : double")
.Input("thetaw : double")
.Input("configw : int64")
.Input("thetao : double")
.Input("configo : int64")
.Input("dt : double")
.Input("h : double")
.Output("grad_s0 : double")
.Output("grad_dporodt : double")
.Output("grad_pt : double")
.Output("grad_perm : double")
.Output("grad_poro : double")
.Output("grad_qw : double")
.Output("grad_qo : double")
.Output("grad_muw : double")
.Output("grad_muo : double")
.Output("grad_sref : double")
.Output("grad_thetaw : double")
.Output("grad_configw : int64")
.Output("grad_thetao : double")
.Output("grad_configo : int64")
.Output("grad_dt : double")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class SaturationNnOp : public OpKernel {
private:
  
public:
  explicit SaturationNnOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(16, context->num_inputs());
    
    
    const Tensor& s0 = context->input(0);
    const Tensor& dporodt = context->input(1);
    const Tensor& pt = context->input(2);
    const Tensor& perm = context->input(3);
    const Tensor& poro = context->input(4);
    const Tensor& qw = context->input(5);
    const Tensor& qo = context->input(6);
    const Tensor& muw = context->input(7);
    const Tensor& muo = context->input(8);
    const Tensor& sref = context->input(9);
    const Tensor& thetaw = context->input(10);
    const Tensor& configw = context->input(11);
    const Tensor& thetao = context->input(12);
    const Tensor& configo = context->input(13);
    const Tensor& dt = context->input(14);
    const Tensor& h = context->input(15);
    
    
    const TensorShape& s0_shape = s0.shape();
    const TensorShape& dporodt_shape = dporodt.shape();
    const TensorShape& pt_shape = pt.shape();
    const TensorShape& perm_shape = perm.shape();
    const TensorShape& poro_shape = poro.shape();
    const TensorShape& qw_shape = qw.shape();
    const TensorShape& qo_shape = qo.shape();
    const TensorShape& muw_shape = muw.shape();
    const TensorShape& muo_shape = muo.shape();
    const TensorShape& sref_shape = sref.shape();
    const TensorShape& thetaw_shape = thetaw.shape();
    const TensorShape& configw_shape = configw.shape();
    const TensorShape& thetao_shape = thetao.shape();
    const TensorShape& configo_shape = configo.shape();
    const TensorShape& dt_shape = dt.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(s0_shape.dims(), 2);
    DCHECK_EQ(dporodt_shape.dims(), 2);
    DCHECK_EQ(pt_shape.dims(), 2);
    DCHECK_EQ(perm_shape.dims(), 2);
    DCHECK_EQ(poro_shape.dims(), 2);
    DCHECK_EQ(qw_shape.dims(), 2);
    DCHECK_EQ(qo_shape.dims(), 2);
    DCHECK_EQ(muw_shape.dims(), 0);
    DCHECK_EQ(muo_shape.dims(), 0);
    DCHECK_EQ(sref_shape.dims(), 2);
    DCHECK_EQ(thetaw_shape.dims(), 1);
    DCHECK_EQ(configw_shape.dims(), 1);
    DCHECK_EQ(thetao_shape.dims(), 1);
    DCHECK_EQ(configo_shape.dims(), 1);
    DCHECK_EQ(dt_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape

    int mw = configw_shape.dim_size(0);
    int mo = configo_shape.dim_size(0);
    
    int nz = s0_shape.dim_size(0), nx = s0_shape.dim_size(1);
    TensorShape sat_shape({nz, nx});
            
    // create output tensor
    
    Tensor* sat = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, sat_shape, &sat));
    
    // get the corresponding Eigen tensors for data access
    
    auto s0_tensor = s0.flat<double>().data();
    auto dporodt_tensor = dporodt.flat<double>().data();
    auto pt_tensor = pt.flat<double>().data();
    auto perm_tensor = perm.flat<double>().data();
    auto poro_tensor = poro.flat<double>().data();
    auto qw_tensor = qw.flat<double>().data();
    auto qo_tensor = qo.flat<double>().data();
    auto muw_tensor = muw.flat<double>().data();
    auto muo_tensor = muo.flat<double>().data();
    auto sref_tensor = sref.flat<double>().data();
    auto thetaw_tensor = thetaw.flat<double>().data();
    auto configw_tensor = configw.flat<int64>().data();
    auto thetao_tensor = thetao.flat<double>().data();
    auto configo_tensor = configo.flat<int64>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto sat_tensor = sat->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    mu_w = *muw_tensor;
    mu_o = *muo_tensor;
    forward(
      thetaw_tensor, configw_tensor, mw, 
      thetao_tensor, configo_tensor, mo, 
      sat_tensor, dporodt_tensor, s0_tensor, pt_tensor, perm_tensor, poro_tensor,
            qw_tensor, qo_tensor, sref_tensor, *dt_tensor, *h_tensor, nz, nx);
  }
};
REGISTER_KERNEL_BUILDER(Name("SaturationNn").Device(DEVICE_CPU), SaturationNnOp);



class SaturationNnGradOp : public OpKernel {
private:
  
public:
  explicit SaturationNnGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_sat = context->input(0);
    const Tensor& sat = context->input(1);
    const Tensor& s0 = context->input(2);
    const Tensor& dporodt = context->input(3);
    const Tensor& pt = context->input(4);
    const Tensor& perm = context->input(5);
    const Tensor& poro = context->input(6);
    const Tensor& qw = context->input(7);
    const Tensor& qo = context->input(8);
    const Tensor& muw = context->input(9);
    const Tensor& muo = context->input(10);
    const Tensor& sref = context->input(11);
    const Tensor& thetaw = context->input(12);
    const Tensor& configw = context->input(13);
    const Tensor& thetao = context->input(14);
    const Tensor& configo = context->input(15);
    const Tensor& dt = context->input(16);
    const Tensor& h = context->input(17);
    
    
    const TensorShape& grad_sat_shape = grad_sat.shape();
    const TensorShape& sat_shape = sat.shape();
    const TensorShape& s0_shape = s0.shape();
    const TensorShape& dporodt_shape = dporodt.shape();
    const TensorShape& pt_shape = pt.shape();
    const TensorShape& perm_shape = perm.shape();
    const TensorShape& poro_shape = poro.shape();
    const TensorShape& qw_shape = qw.shape();
    const TensorShape& qo_shape = qo.shape();
    const TensorShape& muw_shape = muw.shape();
    const TensorShape& muo_shape = muo.shape();
    const TensorShape& sref_shape = sref.shape();
    const TensorShape& thetaw_shape = thetaw.shape();
    const TensorShape& configw_shape = configw.shape();
    const TensorShape& thetao_shape = thetao.shape();
    const TensorShape& configo_shape = configo.shape();
    const TensorShape& dt_shape = dt.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_sat_shape.dims(), 2);
    DCHECK_EQ(sat_shape.dims(), 2);
    DCHECK_EQ(s0_shape.dims(), 2);
    DCHECK_EQ(dporodt_shape.dims(), 2);
    DCHECK_EQ(pt_shape.dims(), 2);
    DCHECK_EQ(perm_shape.dims(), 2);
    DCHECK_EQ(poro_shape.dims(), 2);
    DCHECK_EQ(qw_shape.dims(), 2);
    DCHECK_EQ(qo_shape.dims(), 2);
    DCHECK_EQ(muw_shape.dims(), 0);
    DCHECK_EQ(muo_shape.dims(), 0);
    DCHECK_EQ(sref_shape.dims(), 2);
    DCHECK_EQ(thetaw_shape.dims(), 1);
    DCHECK_EQ(configw_shape.dims(), 1);
    DCHECK_EQ(thetao_shape.dims(), 1);
    DCHECK_EQ(configo_shape.dims(), 1);
    DCHECK_EQ(dt_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
    int mw = configw_shape.dim_size(0);
    int mo = configo_shape.dim_size(0);
    
    int nz = s0_shape.dim_size(0), nx = s0_shape.dim_size(1);
        
    // create output shape
    
    TensorShape grad_s0_shape(s0_shape);
    TensorShape grad_dporodt_shape(dporodt_shape);
    TensorShape grad_pt_shape(pt_shape);
    TensorShape grad_perm_shape(perm_shape);
    TensorShape grad_poro_shape(poro_shape);
    TensorShape grad_qw_shape(qw_shape);
    TensorShape grad_qo_shape(qo_shape);
    TensorShape grad_muw_shape(muw_shape);
    TensorShape grad_muo_shape(muo_shape);
    TensorShape grad_sref_shape(sref_shape);
    TensorShape grad_thetaw_shape(thetaw_shape);
    TensorShape grad_configw_shape(configw_shape);
    TensorShape grad_thetao_shape(thetao_shape);
    TensorShape grad_configo_shape(configo_shape);
    TensorShape grad_dt_shape(dt_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_s0 = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_s0_shape, &grad_s0));
    Tensor* grad_dporodt = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_dporodt_shape, &grad_dporodt));
    Tensor* grad_pt = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_pt_shape, &grad_pt));
    Tensor* grad_perm = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_perm_shape, &grad_perm));
    Tensor* grad_poro = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_poro_shape, &grad_poro));
    Tensor* grad_qw = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_qw_shape, &grad_qw));
    Tensor* grad_qo = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_qo_shape, &grad_qo));
    Tensor* grad_muw = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_muw_shape, &grad_muw));
    Tensor* grad_muo = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_muo_shape, &grad_muo));
    Tensor* grad_sref = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_sref_shape, &grad_sref));
    Tensor* grad_thetaw = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(10, grad_thetaw_shape, &grad_thetaw));
    Tensor* grad_configw = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(11, grad_configw_shape, &grad_configw));
    Tensor* grad_thetao = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(12, grad_thetao_shape, &grad_thetao));
    Tensor* grad_configo = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(13, grad_configo_shape, &grad_configo));
    Tensor* grad_dt = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(14, grad_dt_shape, &grad_dt));
    Tensor* grad_h = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(15, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto s0_tensor = s0.flat<double>().data();
    auto dporodt_tensor = dporodt.flat<double>().data();
    auto pt_tensor = pt.flat<double>().data();
    auto perm_tensor = perm.flat<double>().data();
    auto poro_tensor = poro.flat<double>().data();
    auto qw_tensor = qw.flat<double>().data();
    auto qo_tensor = qo.flat<double>().data();
    auto muw_tensor = muw.flat<double>().data();
    auto muo_tensor = muo.flat<double>().data();
    auto sref_tensor = sref.flat<double>().data();
    auto thetaw_tensor = thetaw.flat<double>().data();
    auto configw_tensor = configw.flat<int64>().data();
    auto thetao_tensor = thetao.flat<double>().data();
    auto configo_tensor = configo.flat<int64>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_sat_tensor = grad_sat.flat<double>().data();
    auto sat_tensor = sat.flat<double>().data();
    auto grad_s0_tensor = grad_s0->flat<double>().data();
    auto grad_dporodt_tensor = grad_dporodt->flat<double>().data();
    auto grad_pt_tensor = grad_pt->flat<double>().data();
    auto grad_perm_tensor = grad_perm->flat<double>().data();
    auto grad_poro_tensor = grad_poro->flat<double>().data();
    auto grad_qw_tensor = grad_qw->flat<double>().data();
    auto grad_qo_tensor = grad_qo->flat<double>().data();
    auto grad_muw_tensor = grad_muw->flat<double>().data();
    auto grad_muo_tensor = grad_muo->flat<double>().data();
    auto grad_sref_tensor = grad_sref->flat<double>().data();
    auto grad_thetaw_tensor = grad_thetaw->flat<double>().data();
    auto grad_thetao_tensor = grad_thetao->flat<double>().data();
    auto grad_dt_tensor = grad_dt->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    mu_w = *muw_tensor;
    mu_o = *muo_tensor;
    backward(
      grad_thetaw_tensor, grad_thetao_tensor,
      thetaw_tensor, configw_tensor, mw, 
      thetao_tensor, configo_tensor, mo, 
      grad_sat_tensor, sat_tensor, s0_tensor, dporodt_tensor, pt_tensor, perm_tensor,
             poro_tensor, qw_tensor, qo_tensor, *dt_tensor, *h_tensor, nz, nx,
             grad_s0_tensor, grad_dporodt_tensor, grad_pt_tensor, grad_perm_tensor,
             grad_poro_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SaturationNnGrad").Device(DEVICE_CPU), SaturationNnGradOp);
