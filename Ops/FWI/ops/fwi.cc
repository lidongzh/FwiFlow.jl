/// \file inner_product.cc
/// \author David Stutz
/// \brief Implementation of a inner product (i.e. fully connected layer)
/// operation in Tensorflow.
#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "ops_utils.h"
using namespace tensorflow;
using namespace std;

REGISTER_OP("FWI")
.Input("model: double")
.Input("index: int64")
.Output("misfit: double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
  shape_inference::ShapeHandle model_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &model_shape));
  shape_inference::ShapeHandle index_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &index_shape));

  c->set_output(0, c->Scalar());
  return Status::OK();
});

/// \brief Implementation of an inner product operation.
/// \param context
/// \author David Stutz
class FwiOp : public OpKernel {
 public:
  /// \brief Constructor.
  /// \param context
  explicit FwiOp(OpKernelConstruction *context) : OpKernel(context) {
  }

  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext *context) override {
#ifdef DEBUG
    cout << "FWIOp--start" << endl;
#endif
    // some checks to be sure ...
    DCHECK_EQ(2, context->num_inputs());

    // get the input tensor
    const Tensor &model = context->input(0);
    const Tensor &index = context->input(1);

    // check shapes of input and weights
    const TensorShape &model_shape = model.shape();
    const TensorShape &index_shape = index.shape();

    // check input is a standing vector
    DCHECK_EQ(model_shape.dims(), 2);
    int nz = model_shape.dim_size(0), nx = model_shape.dim_size(1);
    DCHECK_EQ(index_shape.dims(), 0);

    // // create output shape
    // TensorShape output_shape;
    // output_shape.AddDim(1);
    // output_shape.AddDim(1);

    // create output tensor
    Tensor *misfit = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(), &misfit));

    // get the corresponding Eigen tensors for data access
    auto model_tensor = model.matrix<double>();
    auto index_tensor = index.scalar<int>();
    auto misfit_tensor = misfit->scalar<double>();

    // Setup the phase
    std::string Phase = "../CUFD/Phase" + std::to_string(index_tensor());
    std::string cmd = "cd " + Phase + "/Bin; " + "./CUFD ../Par/Par_file_calc_residual.json ../Par/survey_file.json > out1";
    std::string model_name = Phase + "/Models/Model_Cp.bin";
    std::string para_fname = Phase + "/Par/Par_file_calc_residual.json";

    // Get parameters for padding
    int nz_pad = -1, nx_pad = -1, nPml = -1, nPad = -1;
    // cout << "para_fname = " << para_fname << endl;
    readParJson(para_fname, nz_pad, nx_pad, nPml, nPad);

    // if (nz_pad != (nz + 2 * nPml + nPad) || nx_pad != (nx + 2 * nPml)) {
    //   std::cout << "Dimension Padding Error!" << std::endl;;
    //   exit(1);
    // }

    int index_ = index_tensor();
    float h_l2Obj = -1.0;
    // save input model to binary for FWI calculations (with padding)
    float *h_model = NULL;
    h_model = (float *)malloc(nz_pad * nx_pad * sizeof(float));
    intialArray(h_model, nz_pad * nx_pad, 0.0); // initialize model array



    // int innerJ = 0;
    // for (int i = 0; i < nz_pad; i++) {
    //   for (int j = 0; j < nx_pad; j++) {
    //     innerJ = j - nPml;
    //     if (j < nPml) innerJ  = 0;
    //     if (j >= nx + nPml) innerJ = nx - 1;
    //     if (i >= 0 && i < nPml) {
    //       // cout << "model_" << to_string(i) << "_" << to_string(j) << " = " << model_tensor(0, innerJ) << endl;
    //       h_model[j * nz_pad + i] = model_tensor(0, innerJ);
    //       // h_model[j * nz_pad + i] = 3000.0;
    //     } else if (i >= nPml + nz && i < nz_pad) {
    //       h_model[j * nz_pad + i] = model_tensor(nz - 1, innerJ);
    //       // h_model[j * nz_pad + i] = 3000.0;
    //     } else if (i >= nPml && i < nPml + nz && j >= nPml && j < nPml + nx) {
    //       h_model[j * nz_pad + i] = model_tensor(i - nPml, innerJ);
    //       // h_model[j * nz_pad + i] = 3000.0;
    //     }
    //   }
    // }
    // for (int i = 0; i < nz_pad; i++) {
    //   for (int j = 0; j < nPml; j++) {
    //     h_model[j * nz_pad + i] = h_model[nPml * nz_pad + i];
    //     h_model[(nx_pad - nPml + j) * nz_pad + i] = h_model[(nx_pad - nPml - 1) * nz_pad + i];
    //     // h_model[j * nz_pad + i] = 3000.0;
    //     // h_model[(nx_pad - nPml + j) * nz_pad + i] = 3000.0;
    //   }
    // }
    for (int i = 0; i < nz_pad; i++) {
      for (int j = 0; j < nx_pad; j++) {
        h_model[j * nz_pad + i] = model_tensor(i, j);
      }
    }


    fileBinWrite(h_model, nz_pad * nx_pad, model_name);

    // Call FWI
    if (system(NULL)) {
      // std::cout << "Command = " << cmd << std::endl;
      std::cout << "Computing FWI Forward, Phase = " << std::to_string(index_tensor()) << std::endl;
      system(cmd.c_str());
    } else {
      std::cout << "Command processor doesn't exists" << std::endl;
      exit(1);
    }
    fileBinLoad(&h_l2Obj, 1, Phase + "/Bin/l2Obj.bin");
    misfit_tensor() = h_l2Obj;
    cout << "misfit = " << to_string(h_l2Obj) << endl;

    free(h_model);
#ifdef DEBUG
    cout << "FWIOp--end" << endl;
#endif
  }
};

REGISTER_KERNEL_BUILDER(Name("FWI").Device(DEVICE_CPU), FwiOp);
