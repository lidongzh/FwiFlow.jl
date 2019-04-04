#define d_wavefield(z,x) d_wavefield[(x)*(nz)+(z)]
#define d_Lambda(z,x)  d_Lambda[(x)*(nz)+(z)]
#define d_Cp(z,x)  d_Cp[(x)*(nz)+(z)]
// #define d_data(it,iRec) d_data[(iRec)*(nSteps)+(it)]

__global__ void res_injection(float *d_wavefield, int nz, float *d_res, \
		float *d_Lambda, int it, float dt, int nSteps, int nrec, int *d_z_rec, int *d_x_rec) {

	int iRec = threadIdx.x + blockDim.x*blockIdx.x;
	// float scale = pow(d_Cp(d_z_rec[iRec], d_x_rec[iRec]),2);
	if(iRec >= nrec){
		return;
	}
	// d_wavefield(d_z_rec[iRec], d_x_rec[iRec]) += (d_res[(iRec)*(nSteps)+(it)] + d_res[(iRec)*(nSteps)+(it+1)]) / 2.0 \
	// 		* d_Lambda(d_z_rec[iRec], d_x_rec[iRec]) * dt;
	d_wavefield(d_z_rec[iRec], d_x_rec[iRec]) += d_res[(iRec)*(nSteps)+(it)];
	// d_wavefield(d_z_rec[iRec], d_x_rec[iRec]) += d_res[(iRec)*(nSteps)+(it)] * d_Lambda(d_z_rec[iRec], d_x_rec[iRec]) * dt;
}