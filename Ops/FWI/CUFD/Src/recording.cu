#define d_wavefield(z,x) d_wavefield[(x)*(nz)+(z)]
// #define d_data(it,iRec) d_data[(iRec)*(nSteps)+(it)]

__global__ void recording(float *d_wavefield, int nz, float *d_data, \
		int iShot, int it, int nSteps, int nrec, int *d_z_rec, int *d_x_rec) {

	int iRec = threadIdx.x + blockDim.x*blockIdx.x;
	if(iRec >= nrec){
		return;
	}
	d_data[(iRec)*(nSteps)+(it)] = d_wavefield(d_z_rec[iRec], d_x_rec[iRec]);
}