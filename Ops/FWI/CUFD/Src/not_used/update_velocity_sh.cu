#define d_vx(z,x)  d_vx[(x)*(nz)+(z)]
#define d_vy(z,x)  d_vy[(x)*(nz)+(z)]
#define d_vz(z,x)  d_vz[(x)*(nz)+(z)]
#define d_sxx(z,x) d_sxx[(x)*(nz)+(z)]
#define d_szz(z,x) d_szz[(x)*(nz)+(z)]
#define d_sxz(z,x) d_sxz[(x)*(nz)+(z)]
#define d_mem_dszz_dz(z,x) d_mem_dszz_dz[(x)*(nz)+(z)]
#define d_mem_dsxz_dx(z,x) d_mem_dsxz_dx[(x)*(nz)+(z)]
#define d_mem_dsxz_dz(z,x) d_mem_dsxz_dz[(x)*(nz)+(z)]
#define d_mem_dsxx_dx(z,x) d_mem_dsxx_dx[(x)*(nz)+(z)]
#define d_Lambda(z,x)     d_Lambda[(x)*(nz)+(z)]
#define d_Mu(z,x)         d_Mu[(x)*(nz)+(z)]
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]
#define sh_szz(z,x)  			sh_szz[(x)*(lsizez)+(z)]
#define sh_sxz(z,x)  			sh_sxz[(x)*(lsizez)+(z)]
#define sh_sxx(z,x)  			sh_sxx[(x)*(lsizez)+(z)]
#include<stdio.h>

__global__ void update_velocity_sh(float *d_vz, float *d_vx, float *d_szz, \
	float *d_sxx, float *d_sxz, float *d_mem_dszz_dz, float *d_mem_dsxz_dx, \
	float *d_mem_dsxz_dz, float *d_mem_dsxx_dx, float *d_Lambda, float *d_Mu, \
	float *d_Den, float *d_K_z, float *d_a_z, float *d_b_z, float *d_K_z_half, \
	float *d_a_z_half, float *d_b_z_half, float *d_K_x, float *d_a_x, float *d_b_x, \
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	int nz, int nx, float dt, float dz, float dx){


	// int gidz = blockIdx.x*blockDim.x + threadIdx.x + zerowidth;
 //  int gidx = blockIdx.y*blockDim.y + threadIdx.y + zerowidth;
	// int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  // int gidx = blockIdx.y*blockDim.y + threadIdx.y;
  // printf("gidz=%d, gidx=%d\n", gidz, gidx);

	// shared memory
  int lsizez = blockDim.x + 4;
  int lsizex = blockDim.y + 4;
  int lidz = threadIdx.x + 2;
  int lidx = threadIdx.y + 2;
  int gidz = blockIdx.x*blockDim.x + threadIdx.x + 2;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y + 2;
  // extern __shared__ float sh_vz[];
  // extern __shared__ float sh_vx[];
  __shared__ float sh_szz[720];
  __shared__ float sh_sxx[720];
  __shared__ float sh_sxz[720];
  if(gidz >= nz || gidx >= nx) {return;}
  // printf("lidz=%d, lidx=%d\n", lidz, lidx);
  sh_szz(lidz,lidx) = d_szz(gidz,gidx);
  sh_sxz(lidz,lidx) = d_sxz(gidz,gidx);
  sh_sxx(lidz,lidx) = d_sxx(gidz,gidx);

  if(lidz<4){
  	sh_szz(lidz-2,lidx) = d_szz(gidz-2,gidx);
  	sh_sxz(lidz-2,lidx) = d_sxz(gidz-2,gidx);
  	// sh_sxx(lidz-2,lidx) = d_sxx(gidz-2,gidx);
  }
  if(lidx<4){
    // sh_szz(lidz,lidx-2) = d_szz(gidz,gidx-2);
    sh_sxz(lidz,lidx-2) = d_sxz(gidz,gidx-2);
    sh_sxx(lidz,lidx-2) = d_sxx(gidz,gidx-2);
  }
  // this line should be after loading all left/top edges
  // otherwise, at the boundary, it is possible index 2 or 3
  // are actually >= nz-2 or nx-2
  if(gidz >= nz-2 || gidx >= nx-2) {return;}
  if(lidz>(lsizez-5)){
  	sh_szz(lidz+2,lidx) = d_szz(gidz+2,gidx);
  	sh_sxz(lidz+2,lidx) = d_sxz(gidz+2,gidx);
  	// sh_sxx(lidz+2,lidx) = d_sxx(gidz+2,gidx);
  }
  if(lidx>(lsizex-5)){
  	// sh_szz(lidz,lidx+2) = d_szz(gidz,gidx+2);
  	sh_sxz(lidz,lidx+2) = d_sxz(gidz,gidx+2);
  	sh_sxx(lidz,lidx+2) = d_sxx(gidz,gidx+2);
  }
  __syncthreads();

  // d_szz(gidz,gidx) = sh_szz(lidz,lidx);
  // d_sxz(gidz,gidx) = sh_sxz(lidz,lidx);
  // d_sxx(gidz,gidx) = sh_sxx(lidz,lidx);


  float dszz_dz = 0.0;
  float dsxz_dx = 0.0;
  float dsxz_dz = 0.0;
  float dsxx_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  float buoy_half = 1.0/1000.0;

  // if(gidz>=2 && gidz<=nz-2 && gidx>=2 && gidx<=nx-2) {
  	// printf("gidz=%d, gidx=%d\n", gidz, gidx);
	  // update vz
	  buoy_half = 0.5 * (1.0/d_Den(gidz-1,gidx) + 1.0/d_Den(gidz,gidx));

		dszz_dz = c1*(sh_szz(lidz,lidx)-sh_szz(lidz-1,lidx)) - c2*(sh_szz(lidz+1,lidx)-sh_szz(lidz-2,lidx));
		dsxz_dx = c1*(sh_sxz(lidz,lidx)-sh_sxz(lidz,lidx-1)) - c2*(sh_sxz(lidz,lidx+1)-sh_sxz(lidz,lidx-2));

    if(gidz<32 || gidz>nz-32){
  		d_mem_dszz_dz(gidz,gidx) = d_b_z[gidz]*d_mem_dszz_dz(gidz,gidx) + d_a_z[gidz]*dszz_dz;
        dszz_dz = dszz_dz / d_K_z[gidz] + d_mem_dszz_dz(gidz,gidx);
      }
    if(gidx<32 || gidx>nx-32){
  		d_mem_dsxz_dx(gidz,gidx) = d_b_x[gidx]*d_mem_dsxz_dx(gidz,gidx) + d_a_x[gidx]*dsxz_dx;		
  		dsxz_dx = dsxz_dx / d_K_x[gidx] + d_mem_dsxz_dx(gidz,gidx);
    }

		d_vz(gidz,gidx) = d_vz(gidz,gidx) + (dszz_dz/dz + dsxz_dx/dx) * buoy_half * dt;
	// }
	// else{
	// 	return;
	// }

	// if(gidz>=1 && gidz<=nz-3 && gidx>=1 && gidx<=nx-3) {
		// update vx
		buoy_half = 0.5 * (1.0/d_Den(gidz,gidx) + 1.0/d_Den(gidz,gidx+1));
		dsxz_dz = c1*(sh_sxz(lidz+1,lidx)-sh_sxz(lidz,lidx)) - c2*(sh_sxz(lidz+2,lidx)-sh_sxz(lidz-1,lidx));
		dsxx_dx = c1*(sh_sxx(lidz,lidx+1)-sh_sxx(lidz,lidx)) - c2*(sh_sxx(lidz,lidx+2)-sh_sxx(lidz,lidx-1));

    if(gidz<32 || gidz>nz-32){
  		d_mem_dsxz_dz(gidz,gidx) = d_b_z_half[gidz]*d_mem_dsxz_dz(gidz,gidx) + d_a_z_half[gidz]*dsxz_dz;
      dsxz_dz = dsxz_dz / d_K_z_half[gidz] + d_mem_dsxz_dz(gidz,gidx);
    }

    if(gidx<32 || gidx>nx-32){
  		d_mem_dsxx_dx(gidz,gidx) = d_b_x_half[gidx]*d_mem_dsxx_dx(gidz,gidx) + d_a_x_half[gidx]*dsxx_dx;
  		dsxx_dx = dsxx_dx / d_K_x_half[gidx] + d_mem_dsxx_dx(gidz,gidx);
    }

		d_vx(gidz,gidx) = d_vx(gidz,gidx) + (dsxz_dz/dz + dsxx_dx/dx) * buoy_half * dt;

	// }
	// else{
	// 	return;
	// }


}