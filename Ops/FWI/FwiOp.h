#include <cmath>
#include <iostream>
using namespace std;
// void cufd(double *res, double *d_Cp, const double *Cp, const double *Cs,
//           const double *Den, string dir, int calc_id);

cufd(double *res, double *grad_Cp, double *grad_Cs, double *grad_Den,
     double *grad_stf, const double *Cp, const double *Cs, const double *Den,
     const double *stf, int calc_id, int gpu_id, int group_size, int *shot_ids,
     string para_fname);

void forward(double *res, const double *Cp, const double *Cs, const double *Den,
             string dir) {
  cufd(res, NULL, NULL, NULL, NULL, Cp, Cs, Den, stf, 0, gpu_id, group_size,
       shot_ids, para_fname);
}

void backward(double *d_Cp, const double *Cp, const double *Cs,
              const double *Den, string dir) {
  cufd(NULL, grad_Cp, grad_Cs, grad_Den, grad_stf, Cp, Cs, Den, stf, 1, gpu_id,
       group_size, shot_ids, para_fname);
}