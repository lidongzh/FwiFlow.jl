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
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]


__global__ void update_stress_gl1(float *d_vz, float *d_vx, float *d_szz, \
	float *d_sxx, float *d_mem_dvz_dz, float *d_mem_dvx_dx, float *d_Lambda, \
	float *d_Mu, float *d_K_z_half, float *d_a_z_half, float *d_b_z_half, \
	float *d_K_x, float *d_a_x, float *d_b_x, int nz, int nx, float dt, float dh){

  int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


  float dvz_dz = 0.0;
  float dvx_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  if(gidz>=1 && gidz<=nz-3 && gidx>=2 && gidx<=nx-2) {
	  dvz_dz = c1*(d_vz(gidz+1,gidx)-d_vz(gidz,gidx)) - c2*(d_vz(gidz+2,gidx)-d_vz(gidz-1,gidx));
	  dvx_dx = c1*(d_vx(gidz,gidx)-d_vx(gidz,gidx-1)) - c2*(d_vx(gidz,gidx+1)-d_vx(gidz,gidx-2));

	  if(gidz<32 || gidz>nz-32){
		  d_mem_dvz_dz(gidz,gidx) = d_b_z_half[gidz]*d_mem_dvz_dz(gidz,gidx) + d_a_z_half[gidz]*dvz_dz;
		  dvz_dz = dvz_dz / d_K_z_half[gidz] + d_mem_dvz_dz(gidz,gidx);
		}
		if(gidx<32 || gidx>nx-32){
		  d_mem_dvx_dx(gidz,gidx) = d_b_x[gidx]*d_mem_dvx_dx(gidz,gidx) + d_a_x[gidx]*dvx_dx;
		  dvx_dx = dvx_dx / d_K_x[gidx] + d_mem_dvx_dx(gidz,gidx);
		}

	  d_szz(gidz,gidx) = d_szz(gidz,gidx) + \
	  	((d_Lambda(gidz,gidx)+2.0*d_Mu(gidz,gidx))*dvz_dz + d_Lambda(gidz,gidx)*dvx_dx) * dt/dh;
	  d_sxx(gidz,gidx) = d_sxx(gidz,gidx) + \
	  	(d_Lambda(gidz,gidx)*dvz_dz + (d_Lambda(gidz,gidx)+2.0*d_Mu(gidz,gidx))*dvx_dx) * dt/dh;

  }
  else {
  	return;
  }
}


__global__ void update_stress_gl2(float *d_vz, float *d_vx, float *d_sxz, \
	float *d_mem_dvz_dx, float *d_mem_dvx_dz, float *d_Mu, float *d_K_z, \
	float *d_a_z, float *d_b_z, float *d_K_x_half, float *d_a_x_half, \
	float *d_b_x_half, int nz, int nx, float dt, float dh){


	int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  float mu_ave = 0.0;
	float dvx_dz = 0.0;
	float dvz_dx = 0.0;

  if(gidz>=2 && gidz<=nz-2 && gidx>=1 && gidx<=nx-3){


	  dvx_dz = c1*(d_vx(gidz,gidx)-d_vx(gidz-1,gidx)) - c2*(d_vx(gidz+1,gidx)-d_vx(gidz-2,gidx));
	  dvz_dx = c1*(d_vz(gidz,gidx+1)-d_vz(gidz,gidx)) - c2*(d_vz(gidz,gidx+2)-d_vz(gidz,gidx-1));

	  if(gidz<32 || gidz>nz-32){
		  d_mem_dvx_dz(gidz,gidx) = d_b_z[gidz]*d_mem_dvx_dz(gidz,gidx) + d_a_z[gidz]*dvx_dz;
		  dvx_dz = dvx_dz / d_K_z[gidz] + d_mem_dvx_dz(gidz,gidx);
		}
		if(gidx<32 || gidx>nx-32){
		  d_mem_dvz_dx(gidz,gidx) = d_b_x_half[gidx]*d_mem_dvz_dx(gidz,gidx) + d_a_x_half[gidx]*dvz_dx;
		  dvz_dx = dvz_dx / d_K_x_half[gidx] + d_mem_dvz_dx(gidz,gidx);
		}

	  d_sxz(gidz,gidx) = d_sxz(gidz,gidx) + mu_ave * (dvx_dz + dvz_dx) * dt/dh;
	}
	else{
		return;
	}

}


void update_stress_gl(float *d_vz, float *d_vx, float *d_szz, \
	float *d_sxx, float *d_sxz, float *d_mem_dvz_dz, float *d_mem_dvz_dx, \
	float *d_mem_dvx_dz, float *d_mem_dvx_dx, float *d_Lambda, float *d_Mu, \
	float *d_Den, float *d_K_z, float *d_a_z, float *d_b_z, float *d_K_z_half, \
	float *d_a_z_half, float *d_b_z_half, float *d_K_x, float *d_a_x, float *d_b_x, \
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	int nz, int nx, float dt, float dh, dim3 blocks, dim3 threads){


	update_stress_gl1<<<blocks,threads>>>(d_vz, d_vx, d_szz, d_sxx, d_mem_dvz_dz, \
		d_mem_dvx_dx, d_Lambda, d_Mu, d_K_z_half, d_a_z_half, d_b_z_half, d_K_x, d_a_x, \
		d_b_x, nz, nx, dt, dh);

	update_stress_gl2<<<blocks,threads>>>(d_vz, d_vx, d_sxz, d_mem_dvz_dx, d_mem_dvx_dz, \
		d_Mu, d_K_z, d_a_z, d_b_z, d_K_x_half, d_a_x_half, d_b_x_half, nz, nx, dt, dh);

}