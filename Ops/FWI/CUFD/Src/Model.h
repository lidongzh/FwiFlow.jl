// Dongzhuo Li 04/20/2018
#ifndef MODEL_H__
#define MODEL_H__

#include "Parameter.h"

// void fileBinLoad(float *h_bin, int size, std::string fname);
// void intialArray(float *ip, int size, float value);
// __global__ void moduliInit(float *d_Cp, float *d_Cs, float *d_Den, float *d_Lambda, float *d_Mu, int nx, int ny);

class Model {

  private:
    int nz_, nx_;

  public:
    Model();
    Model(const Parameter &para);
    ~Model();

    float *h_Cp;
    float *h_Cs;
    float *h_Den;
    float *d_Cp;
    float *d_Cs;
    float *d_Den;

    float *h_Lambda;
    float *h_Mu;
    float *d_Lambda;
    float *d_Mu;
    float *d_ave_Mu;
    float *d_ave_Byc_a;
    float *d_ave_Byc_b;

    float *h_CpGrad;
    float *d_CpGrad;

    int nz() {
        return nz_;
    }
    int nx() {
        return nx_;
    }
};

#endif
