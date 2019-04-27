#include <iostream>
#include <string>
#include "Model.h"
#include "Parameter.h"
#include "utilities.h"


// model default constructor
Model::Model() {
    nz_ = 1000;
    nx_ = 1000;
    dim3 threads(TX, TY);
    dim3 blocks((nz_ + TX - 1) / TX, (nx_ + TY - 1) / TY);

    h_Cp = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Cs = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Den = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Lambda = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Mu = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_CpGrad = (float *)malloc(nz_ * nx_ * sizeof(float));

    initialArray(h_Cp, nz_ * nx_, 3300.0);
    // initialArray(h_Cs, nz*nx, 3300.0/sqrt(3.0));
    initialArray(h_Cs, nz_ * nx_, 0.0);
    initialArray(h_Den, nz_ * nx_, 1000.0);
    initialArray(h_Lambda, nz_ * nx_, 0.0);
    initialArray(h_Mu, nz_ * nx_, 0.0);
    initialArray(h_CpGrad, nz_ * nx_, 0.0);


    CHECK(cudaMalloc((void **)&d_Cp, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Cs, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Den, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Lambda, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Mu, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_ave_Mu, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_ave_Byc_a, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_ave_Byc_b, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_CpGrad, nz_ * nx_ * sizeof(float)));
    intialArrayGPU<<<blocks,threads>>>(d_ave_Mu, nz_, nx_, 0.0);
    intialArrayGPU<<<blocks,threads>>>(d_CpGrad, nz_, nx_, 0.0);
    intialArrayGPU<<<blocks,threads>>>(d_ave_Byc_a, nz_, nx_, 1.0/1000.0);
    intialArrayGPU<<<blocks,threads>>>(d_ave_Byc_b, nz_, nx_, 1.0/1000.0);

    CHECK(cudaMemcpy(d_Cp, h_Cp, nz_ * nx_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Cs, h_Cs, nz_ * nx_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Den, h_Den, nz_ * nx_ * sizeof(float), cudaMemcpyHostToDevice));

    moduliInit<<< blocks,threads>>>(d_Cp, d_Cs, d_Den, d_Lambda, d_Mu, nz_, nx_);
    aveMuInit<<<blocks,threads>>>(d_Mu, d_ave_Mu, nz_, nx_);
    aveBycInit<<<blocks,threads>>>(d_Den, d_ave_Byc_a, d_ave_Byc_b, nz_, nx_);

    CHECK(cudaMemcpy(h_Lambda, d_Lambda, nz_ * nx_ * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_Mu, d_Mu, nz_ * nx_ * sizeof(float), cudaMemcpyDeviceToHost));


}

// model constructor from parameter file
Model::Model(const Parameter &para, const float *Cp_, const float*Cs_, const float *Den_) {

    nz_ = para.nz();
    nx_ = para.nx();

    dim3 threads(32, 16);
    dim3 blocks((nz_ + 32 - 1) / 32, (nx_ + 16 - 1) / 16);

    h_Cp = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Cs = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Den = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Lambda = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Mu = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_CpGrad = (float *)malloc(nz_ * nx_ * sizeof(float));

    // load Vp, Vs, and Den binaries
    if (para.Cp_fname() == para.Cs_fname()) {
        // only load vp
        fileBinLoad(h_Cp, nz_ * nx_, para.Cp_fname());
        initialArray(h_Cs, nz_ * nx_, 0.0);
        initialArray(h_Den, nz_ * nx_, 1000.0);
    } else {
        for(int i=0;i< nz_*nx_;i++){
            h_Cp[i] = Cp_[i];
            h_Cs[i] = Cs_[i];
            h_Den[i] = Den_[i];
        }
        #if 0
        fileBinLoad(h_Cp, nz_ * nx_, para.Cp_fname());
        fileBinLoad(h_Cs, nz_ * nx_, para.Cs_fname());
        fileBinLoad(h_Den, nz_ * nx_, para.Den_fname());
        #endif
    }
    initialArray(h_Lambda, nz_ * nx_, 0.0);
    initialArray(h_Mu, nz_ * nx_, 0.0);
    initialArray(h_CpGrad, nz_ * nx_, 0.0);


    CHECK(cudaMalloc((void **)&d_Cp, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Cs, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Den, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Lambda, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Mu, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_ave_Mu, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_ave_Byc_a, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_ave_Byc_b, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_CpGrad, nz_ * nx_ * sizeof(float)));
    intialArrayGPU<<<blocks,threads>>>(d_ave_Mu, nz_, nx_, 0.0);
    intialArrayGPU<<<blocks,threads>>>(d_CpGrad, nz_, nx_, 0.0);
    intialArrayGPU<<<blocks,threads>>>(d_ave_Byc_a, nz_, nx_, 1.0/1000.0);
    intialArrayGPU<<<blocks,threads>>>(d_ave_Byc_b, nz_, nx_, 1.0/1000.0);

    CHECK(cudaMemcpy(d_Cp, h_Cp, nz_ * nx_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Cs, h_Cs, nz_ * nx_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Den, h_Den, nz_ * nx_ * sizeof(float), cudaMemcpyHostToDevice));

    moduliInit<<<blocks,threads>>>(d_Cp, d_Cs, d_Den, d_Lambda, d_Mu, nz_, nx_);
    aveMuInit<<<blocks,threads>>>(d_Mu, d_ave_Mu, nz_, nx_);
    aveBycInit<<<blocks,threads>>>(d_Den, d_ave_Byc_a, d_ave_Byc_b, nz_, nx_);

    CHECK(cudaMemcpy(h_Lambda, d_Lambda, nz_ * nx_ * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_Mu, d_Mu, nz_ * nx_ * sizeof(float), cudaMemcpyDeviceToHost));

}
// model constructor from parameter file
Model::Model(const Parameter &para) {

    nz_ = para.nz();
    nx_ = para.nx();

    dim3 threads(32, 16);
    dim3 blocks((nz_ + 32 - 1) / 32, (nx_ + 16 - 1) / 16);

    h_Cp = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Cs = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Den = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Lambda = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_Mu = (float *)malloc(nz_ * nx_ * sizeof(float));
    h_CpGrad = (float *)malloc(nz_ * nx_ * sizeof(float));

    // load Vp, Vs, and Den binaries
    if (para.Cp_fname() == para.Cs_fname()) {
        // only load vp
        fileBinLoad(h_Cp, nz_ * nx_, para.Cp_fname());
        initialArray(h_Cs, nz_ * nx_, 0.0);
        initialArray(h_Den, nz_ * nx_, 1000.0);
    } else {
        fileBinLoad(h_Cp, nz_ * nx_, para.Cp_fname());
        fileBinLoad(h_Cs, nz_ * nx_, para.Cs_fname());
        fileBinLoad(h_Den, nz_ * nx_, para.Den_fname());
    }
    initialArray(h_Lambda, nz_ * nx_, 0.0);
    initialArray(h_Mu, nz_ * nx_, 0.0);
    initialArray(h_CpGrad, nz_ * nx_, 0.0);


    CHECK(cudaMalloc((void **)&d_Cp, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Cs, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Den, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Lambda, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Mu, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_ave_Mu, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_ave_Byc_a, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_ave_Byc_b, nz_ * nx_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_CpGrad, nz_ * nx_ * sizeof(float)));
    intialArrayGPU<<<blocks,threads>>>(d_ave_Mu, nz_, nx_, 0.0);
    intialArrayGPU<<<blocks,threads>>>(d_CpGrad, nz_, nx_, 0.0);
    intialArrayGPU<<<blocks,threads>>>(d_ave_Byc_a, nz_, nx_, 1.0/1000.0);
    intialArrayGPU<<<blocks,threads>>>(d_ave_Byc_b, nz_, nx_, 1.0/1000.0);

    CHECK(cudaMemcpy(d_Cp, h_Cp, nz_ * nx_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Cs, h_Cs, nz_ * nx_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Den, h_Den, nz_ * nx_ * sizeof(float), cudaMemcpyHostToDevice));

    moduliInit<<<blocks,threads>>>(d_Cp, d_Cs, d_Den, d_Lambda, d_Mu, nz_, nx_);
    aveMuInit<<<blocks,threads>>>(d_Mu, d_ave_Mu, nz_, nx_);
    aveBycInit<<<blocks,threads>>>(d_Den, d_ave_Byc_a, d_ave_Byc_b, nz_, nx_);

    CHECK(cudaMemcpy(h_Lambda, d_Lambda, nz_ * nx_ * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_Mu, d_Mu, nz_ * nx_ * sizeof(float), cudaMemcpyDeviceToHost));

}


Model::~Model() {
    free(h_Cp);
    free(h_Cs);
    free(h_Den);
    free(h_Lambda);
    free(h_Mu);
    free(h_CpGrad);
    CHECK(cudaFree(d_Cp));
    CHECK(cudaFree(d_Cs));
    CHECK(cudaFree(d_Den));
    CHECK(cudaFree(d_Lambda));
    CHECK(cudaFree(d_Mu));
    CHECK(cudaFree(d_ave_Mu));
    CHECK(cudaFree(d_ave_Byc_a));
    CHECK(cudaFree(d_ave_Byc_b));
    CHECK(cudaFree(d_CpGrad));
}