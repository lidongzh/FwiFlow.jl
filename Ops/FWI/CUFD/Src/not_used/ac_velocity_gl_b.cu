#define d_vx(z,x)  d_vx[(x)*(nz)+(z)]
#define d_vy(z,x)  d_vy[(x)*(nz)+(z)]
#define d_vz(z,x)  d_vz[(x)*(nz)+(z)]
#define d_szz(z,x) d_szz[(x)*(nz)+(z)] // Pressure
#define d_mem_dszz_dz(z,x) d_mem_dszz_dz[(x)*(nz)+(z)]
#define d_mem_dsxx_dx(z,x) d_mem_dsxx_dx[(x)*(nz)+(z)]
#define d_Lambda(z,x)     d_Lambda[(x)*(nz)+(z)]
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]
#include<stdio.h>

__global__ void ac_velocity_gl_b(float *d_vz, float *d_vx, float *d_szz, \
	float *d_mem_dszz_dz, float *d_mem_dsxx_dx, float *d_Lambda, \
	float *d_Den, float *d_K_z, float *d_a_z, float *d_b_z, \
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	int nz, int nx, float dt, float dz, float dx, int nPml, int nPad){


	int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


  float dszz_dz = 0.0;
  float dsxx_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  float buoy_half = 1.0/1000.0;

  if(gidz>=2+nPml && gidz<=nz-nPad-3-nPml && gidx>=2+nPml && gidx<=nx-3-nPml) {

	  // update vz
	  buoy_half = 0.5 * (1.0/d_Den(gidz-1,gidx) + 1.0/d_Den(gidz,gidx));

		dszz_dz = c1*(d_szz(gidz,gidx)-d_szz(gidz-1,gidx)) - c2*(d_szz(gidz+1,gidx)-d_szz(gidz-2,gidx));

		d_vz(gidz,gidx) = d_vz(gidz,gidx) - dszz_dz/dz * buoy_half * dt;


		// update vx
		buoy_half = 0.5 * (1.0/d_Den(gidz,gidx) + 1.0/d_Den(gidz,gidx+1));
		dsxx_dx = c1*(d_szz(gidz,gidx+1)-d_szz(gidz,gidx)) - c2*(d_szz(gidz,gidx+2)-d_szz(gidz,gidx-1));

		d_vx(gidz,gidx) = d_vx(gidz,gidx) - dsxx_dx/dx * buoy_half * dt;

	}

	else {
		return;
	}


}