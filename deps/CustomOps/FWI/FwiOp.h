#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include "Src/Parameter.h"
#include <fstream>
#include <vector>
using namespace std;
// void cufd(double *misfit, double *d_Lambda, const double *Lambda, const double *Mu,
//           const double *Den, string dir, int calc_id);

void cufd(double *misfit, double *grad_Lambda, double *grad_Mu, double *grad_Den,
          double *grad_stf, const double *Lambda, const double *Mu,
          const double *Den, const double *stf, int calc_id, const int gpu_id,
          int group_size, const int *shot_ids, const string para_fname);

void forward(double *misfit, const double *Lambda, const double *Mu,
             const double *Den, const double *stf, const int gpu_id,
             int group_size, const int *shot_ids, const string para_fname) {
  cufd(misfit, NULL, NULL, NULL, NULL, Lambda, Mu, Den, stf, 0, gpu_id, group_size,
       shot_ids, para_fname);
}

void backward(double *grad_Lambda, double *grad_Mu, double *grad_Den,
              double *grad_stf, const double *Lambda, const double *Mu,
              const double *Den, const double *stf, const int gpu_id,
              int group_size, const int *shot_ids, const string para_fname) {
  cufd(NULL, grad_Lambda, grad_Mu, grad_Den, grad_stf, Lambda, Mu, Den, stf, 1, gpu_id,
       group_size, shot_ids, para_fname);
}

void obscalc(const double *Lambda, const double *Mu,
             const double *Den, const double *stf, const int gpu_id,
             int group_size, const int *shot_ids, const string para_fname,
             OpKernelContext* context) {
  double dummy;
  cufd(&dummy, NULL, NULL, NULL, NULL, Lambda, Mu, Den, stf, 2, gpu_id, group_size,
       shot_ids, para_fname);
  Parameter para(para_fname, 2);

  std::vector<float> dvec; 
  for (int j = 0; j< group_size; j++){
       
     std::string filename = para.data_dir_name() + "/Shot" +
                         std::to_string(shot_ids[0]) + ".bin";
     printf("Processing file %s ... ", filename.c_str());
     float f; int cnt;
     std::ifstream fin("male_16_down.bin", std::ios::binary);
     while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))){
          cnt += 1;
          dvec.push_back(f);
     }
     printf(" read %d floats\n", cnt);
     fin.close();
          
  }
  
  TensorShape misfit_shape({dvec.size()});
  Tensor* misfit = NULL;
  OP_REQUIRES_OK(context, context->allocate_output(0, misfit_shape, &misfit));
  auto misfit_tensor = misfit->flat<double>().data();
  for(int i=0;i<dvec.size();i++) misfit_tensor[i] = (double) dvec[i];

}