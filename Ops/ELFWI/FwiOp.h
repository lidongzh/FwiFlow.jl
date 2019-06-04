#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
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

void obscalc(double *misfit, const double *Lambda, const double *Mu,
             const double *Den, const double *stf, const int gpu_id,
             int group_size, const int *shot_ids, const string para_fname) {
  cufd(misfit, NULL, NULL, NULL, NULL, Lambda, Mu, Den, stf, 2, gpu_id, group_size,
       shot_ids, para_fname);
}