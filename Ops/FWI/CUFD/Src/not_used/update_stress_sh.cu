#define d_vx(z,x)  d_vx[(x)*(nz)+(z)]
#define d_vy(z,x)  d_vy[(x)*(nz)+(z)]
#define d_vz(z,x)  d_vz[(x)*(nz)+(z)]
#define d_sxx(z,x) d_sxx[(x)*(nz)+(z)]
#define d_szz(z,x) d_szz[(x)*(nz)+(z)]
#define d_sxz(z,x) d_sxz[(x)*(nz)+(z)]
#define d_mem_dvz_dz(z,x) d_mem_dvz_dz[(x)*(nz)+(z)]
#define d_mem_dvz_dx(z,x) d_mem_dvz_dx[(x)*(nz)+(z)]
#define d_mem_dvx_dz(z,x) d_mem_dvx_dz[(x)*(nz)+(z)]
#define d_mem_dvx_dx(z,x) d_mem_dvx_dx[(x)*(nz)+(z)]
#define d_Lambda(z,x)     d_Lambda[(x)*(nz)+(z)]
#define d_Mu(z,x)         d_Mu[(x)*(nz)+(z)]
#define d_ave_Mu(z,x)     d_ave_Mu[(x)*(nz)+(z)]
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]
#define sh_vz(z,x)  			sh_vz[(x)*(lsizez)+(z)]
#define sh_vx(z,x)  			sh_vx[(x)*(lsizez)+(z)]
#include<stdio.h>

__global__ void update_stress_sh(float *d_vz, float *d_vx, float *d_szz, \
	float *d_sxx, float *d_sxz, float *d_mem_dvz_dz, float *d_mem_dvz_dx, \
	float *d_mem_dvx_dz, float *d_mem_dvx_dx, float *d_Lambda, float *d_Mu, float *d_ave_Mu,\
	float *d_Den, float *d_K_z, float *d_a_z, float *d_b_z, float *d_K_z_half, \
	float *d_a_z_half, float *d_b_z_half, float *d_K_x, float *d_a_x, float *d_b_x, \
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	int nz, int nx, float dt, float dz, float dx){

	// int gidz = blockIdx.x*blockDim.x + threadIdx.x + zerowidth;
  // int gidx = blockIdx.y*blockDim.y + threadIdx.y + zerowidth;
  // int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  // int gidx = blockIdx.y*blockDim.y + threadIdx.y;

	// shared memory
  int lsizez = blockDim.x + 4;
  int lsizex = blockDim.y + 4;
  int lidz = threadIdx.x + 2;
  int lidx = threadIdx.y + 2;
  int gidz = blockIdx.x*blockDim.x + threadIdx.x + 2;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y + 2;
  // extern __shared__ float sh_vz[];
  // extern __shared__ float sh_vx[];
  __shared__ float sh_vz[720];
  __shared__ float sh_vx[720];
  if(gidz >= nz || gidx >= nx) {return;}
  // printf("lidz=%d, lidx=%d\n", lidz, lidx);
  sh_vz(lidz,lidx) = d_vz(gidz,gidx);
  sh_vx(lidz,lidx) = d_vx(gidz,gidx);
  if(lidz<4){
  	sh_vz(lidz-2,lidx) = d_vz(gidz-2,gidx);
  	sh_vx(lidz-2,lidx) = d_vx(gidz-2,gidx);
  }
  if(lidx<4){
  	sh_vz(lidz,lidx-2) = d_vz(gidz,gidx-2);
  	sh_vx(lidz,lidx-2) = d_vx(gidz,gidx-2);
  }
  if(gidz >= nz-2 || gidx >= nx-2) {return;}
  if(lidz>(lsizez-5)){
  	sh_vz(lidz+2,lidx) = d_vz(gidz+2,gidx);
  	sh_vx(lidz+2,lidx) = d_vx(gidz+2,gidx);
  }
  if(lidx>(lsizex-5)){
  	sh_vz(lidz,lidx+2) = d_vz(gidz,gidx+2);
  	sh_vx(lidz,lidx+2) = d_vx(gidz,gidx+2);
  }
  __syncthreads();
  // printf("lidz=%d, lidx=%d\n", lidz, lidx);




  float dvz_dz = 0.0;
  float dvx_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  float lambda = 0.0;
  float mu = 0.0;

  // if(gidz>=1 && gidz<=nz-3 && gidx>=2 && gidx<=nx-2) {
	  // dvz_dz = c1*(d_vz(gidz+1,gidx)-d_vz(gidz,gidx)) - c2*(d_vz(gidz+2,gidx)-d_vz(gidz-1,gidx));
	  // dvx_dx = c1*(d_vx(gidz,gidx)-d_vx(gidz,gidx-1)) - c2*(d_vx(gidz,gidx+1)-d_vx(gidz,gidx-2));
	  dvz_dz = c1*(sh_vz(lidz+1,lidx)-sh_vz(lidz,lidx)) - c2*(sh_vz(lidz+2,lidx)-sh_vz(lidz-1,lidx));
	  dvx_dx = c1*(sh_vx(lidz,lidx)-sh_vx(lidz,lidx-1)) - c2*(sh_vx(lidz,lidx+1)-sh_vx(lidz,lidx-2));
  	// dvz_dz = c1*(sh_vz(lidz+1,lidx)-sh_vz(lidz,lidx)) - c2*(sh_vz(lidz+2,lidx)-sh_vz(lidz-1,lidx));
	  // dvx_dx = c1*(sh_vx(lidz,lidx)-sh_vx(lidz,lidx-1)) - c2*(sh_vx(lidz,lidx+1)-sh_vx(lidz,lidx-2));

	  if(gidz<32 || gidz>nz-32){
	  	d_mem_dvz_dz(gidz,gidx) = d_b_z_half[gidz]*d_mem_dvz_dz(gidz,gidx) + d_a_z_half[gidz]*dvz_dz;
	  	dvz_dz = dvz_dz / d_K_z_half[gidz] + d_mem_dvz_dz(gidz,gidx);
		}
		if(gidx<32 || gidx>nx-32){
		  d_mem_dvx_dx(gidz,gidx) = d_b_x[gidx]*d_mem_dvx_dx(gidz,gidx) + d_a_x[gidx]*dvx_dx;
		  dvx_dx = dvx_dx / d_K_x[gidx] + d_mem_dvx_dx(gidz,gidx);
		}


    lambda = d_Lambda(gidz,gidx);
    mu = d_Mu(gidz,gidx);

    d_szz(gidz,gidx) = d_szz(gidz,gidx) + \
      ((lambda+2.0*mu)*dvz_dz/dz + lambda*dvx_dx/dx) * dt;
    d_sxx(gidz,gidx) = d_sxx(gidz,gidx) + \
      (lambda*dvz_dz/dz + (lambda+2.0*mu)*dvx_dx/dx) * dt;

  // }
  // else {
  // 	return;
  // }


  // if(gidz>=2 && gidz<=nz-2 && gidx>=1 && gidx<=nx-3){
	  float dvx_dz = 0.0;
	  float dvz_dx = 0.0;

	  // dvx_dz = c1*(d_vx(gidz,gidx)-d_vx(gidz-1,gidx)) - c2*(d_vx(gidz+1,gidx)-d_vx(gidz-2,gidx));
	  // dvz_dx = c1*(d_vz(gidz,gidx+1)-d_vz(gidz,gidx)) - c2*(d_vz(gidz,gidx+2)-d_vz(gidz,gidx-1));
	  dvx_dz = c1*(sh_vx(lidz,lidx)-sh_vx(lidz-1,lidx)) - c2*(sh_vx(lidz+1,lidx)-sh_vx(lidz-2,lidx));
	  dvz_dx = c1*(sh_vz(lidz,lidx+1)-sh_vz(lidz,lidx)) - c2*(sh_vz(lidz,lidx+2)-sh_vz(lidz,lidx-1));
	  // dvx_dz = c1*(sh_vx(lidz,lidx)-sh_vx(lidz-1,lidx)) - c2*(sh_vx(lidz+1,lidx)-sh_vx(lidz-2,lidx));
	  // dvz_dx = c1*(sh_vz(lidz,lidx+1)-sh_vz(lidz,lidx)) - c2*(sh_vz(lidz,lidx+2)-sh_vz(lidz,lidx-1));
	  if(gidz<32 || gidz>nz-32){
		  d_mem_dvx_dz(gidz,gidx) = d_b_z[gidz]*d_mem_dvx_dz(gidz,gidx) + d_a_z[gidz]*dvx_dz;
		  dvx_dz = dvx_dz / d_K_z[gidz] + d_mem_dvx_dz(gidz,gidx);
		}
		if(gidx<32 || gidx>nx-32){
		  d_mem_dvz_dx(gidz,gidx) = d_b_x_half[gidx]*d_mem_dvz_dx(gidz,gidx) + d_a_x_half[gidx]*dvz_dx;  
	  	dvz_dx = dvz_dx / d_K_x_half[gidx] + d_mem_dvz_dx(gidz,gidx);
	  }

	  d_sxz(gidz,gidx) = d_sxz(gidz,gidx) + d_ave_Mu(gidz,gidx) * (dvx_dz/dz + dvz_dx/dx) * dt;
	// }
	// else{
	// 	return;
	// }

}