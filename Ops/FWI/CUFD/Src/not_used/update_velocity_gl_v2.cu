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
#include<stdio.h>

__global__ void update_velocity_gl1(float *d_vz, float *d_szz, \
	float *d_sxz, float *d_mem_dszz_dz, float *d_mem_dsxz_dx, \
	float *d_Den, float *d_K_z, float *d_a_z, float *d_b_z, float *d_K_x, \
	float *d_a_x, float *d_b_x, int nz, int nx, float dt, float dh){


	int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


  float dszz_dz = 0.0;
  float dsxz_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  float buoy_half = 1.0/1000.0;

  if(gidz>=2 && gidz<=nz-2 && gidx>=2 && gidx<=nx-2) {
  	// printf("gidz=%d, gidx=%d\n", gidz, gidx);
	  // update vz
	  buoy_half = 0.5 * (1.0/d_Den(gidz-1,gidx) + 1.0/d_Den(gidz,gidx));

		dszz_dz = c1*(d_szz(gidz,gidx)-d_szz(gidz-1,gidx)) - c2*(d_szz(gidz+1,gidx)-d_szz(gidz-2,gidx));
		dsxz_dx = c1*(d_sxz(gidz,gidx)-d_sxz(gidz,gidx-1)) - c2*(d_sxz(gidz,gidx+1)-d_sxz(gidz,gidx-2));

		if(gidz<32 || gidz>nz-32){
			d_mem_dszz_dz(gidz,gidx) = d_b_z[gidz]*d_mem_dszz_dz(gidz,gidx) + d_a_z[gidz]*dszz_dz;
			dszz_dz = dszz_dz / d_K_z[gidz] + d_mem_dszz_dz(gidz,gidx);
	  }
	  if(gidx<32 || gidx>nx-32){
			d_mem_dsxz_dx(gidz,gidx) = d_b_x[gidx]*d_mem_dsxz_dx(gidz,gidx) + d_a_x[gidx]*dsxz_dx;
			dsxz_dx = dsxz_dx / d_K_x[gidx] + d_mem_dsxz_dx(gidz,gidx);
		}

		d_vz(gidz,gidx) = d_vz(gidz,gidx) + (dszz_dz + dsxz_dx) * buoy_half * dt/dh;
	}
	else{
		return;
	}
}


__global__ void update_velocity_gl2(float *d_vx, float *d_sxx, float *d_sxz, \
	float *d_mem_dsxz_dz, float *d_mem_dsxx_dx, float *d_Den, float *d_K_z_half, \
	float *d_a_z_half, float *d_b_z_half, float *d_K_x_half, float *d_a_x_half, \
	float *d_b_x_half, int nz, int nx, float dt, float dh){


	int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;

  float dsxz_dz = 0.0;
  float dsxx_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  float buoy_half = 1.0/1000.0;

	if(gidz>=1 && gidz<=nz-3 && gidx>=1 && gidx<=nx-3) {
		// update vx
		buoy_half = 0.5 * (1.0/d_Den(gidz,gidx) + 1.0/d_Den(gidz,gidx+1));
		dsxz_dz = c1*(d_sxz(gidz+1,gidx)-d_sxz(gidz,gidx)) - c2*(d_sxz(gidz+2,gidx)-d_sxz(gidz-1,gidx));
		dsxx_dx = c1*(d_sxx(gidz,gidx+1)-d_sxx(gidz,gidx)) - c2*(d_sxx(gidz,gidx+2)-d_sxx(gidz,gidx-1));

		if(gidz<32 || gidz>nz-32){
			d_mem_dsxz_dz(gidz,gidx) = d_b_z_half[gidz]*d_mem_dsxz_dz(gidz,gidx) + d_a_z_half[gidz]*dsxz_dz;
			dsxz_dz = dsxz_dz / d_K_z_half[gidz] + d_mem_dsxz_dz(gidz,gidx);
		}
		if(gidx<32 || gidx>nx-32){
			d_mem_dsxx_dx(gidz,gidx) = d_b_x_half[gidx]*d_mem_dsxx_dx(gidz,gidx) + d_a_x_half[gidx]*dsxx_dx;	
			dsxx_dx = dsxx_dx / d_K_x_half[gidx] + d_mem_dsxx_dx(gidz,gidx);
		}

		d_vx(gidz,gidx) = d_vx(gidz,gidx) + (dsxz_dz + dsxx_dx) * buoy_half * dt/dh;

	}
	else{
		return;
	}
}



void update_velocity_gl(float *d_vz, float *d_vx, float *d_szz, \
	float *d_sxx, float *d_sxz, float *d_mem_dszz_dz, float *d_mem_dsxz_dx, \
	float *d_mem_dsxz_dz, float *d_mem_dsxx_dx, float *d_Lambda, float *d_Mu, \
	float *d_Den, float *d_K_z, float *d_a_z, float *d_b_z, float *d_K_z_half, \
	float *d_a_z_half, float *d_b_z_half, float *d_K_x, float *d_a_x, float *d_b_x, \
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	int nz, int nx, float dt, float dh, dim3 blocks, dim3 threads){


	update_velocity_gl1<<<blocks,threads>>>(d_vz, d_szz, d_sxz, d_mem_dszz_dz, d_mem_dsxz_dx, \
		d_Den, d_K_z, d_a_z, d_b_z, d_K_x, d_a_x, d_b_x, nz, nx, dt, dh);


	update_velocity_gl2<<<blocks,threads>>>(d_vx, d_sxx, d_sxz, d_mem_dsxz_dz, d_mem_dsxx_dx, \
		d_Den, d_K_z_half, d_a_z_half, d_b_z_half, d_K_x_half, d_a_x_half, d_b_x_half, nz, nx, dt, dh);



}