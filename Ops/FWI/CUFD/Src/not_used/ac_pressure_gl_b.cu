#define d_vx(z,x)  d_vx[(x)*(nz)+(z)]
#define d_vy(z,x)  d_vy[(x)*(nz)+(z)]
#define d_vz(z,x)  d_vz[(x)*(nz)+(z)]
#define d_szz(z,x) d_szz[(x)*(nz)+(z)] // Pressure
#define d_mem_dvz_dz(z,x) d_mem_dvz_dz[(x)*(nz)+(z)]
#define d_mem_dvx_dx(z,x) d_mem_dvx_dx[(x)*(nz)+(z)]
#define d_Lambda(z,x)     d_Lambda[(x)*(nz)+(z)]
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]

__global__ void ac_pressure_gl_b(float *d_vz, float *d_vx, float *d_szz, \
	float *d_mem_dvz_dz, float *d_mem_dvx_dx, float *d_Lambda, \
	float *d_Den, float *d_K_z_half, float *d_a_z_half, float *d_b_z_half, \
	float *d_K_x, float *d_a_x, float *d_b_x, \
	int nz, int nx, float dt, float dz, float dx, int nPml, int nPad){

  int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


  float dvz_dz = 0.0;
  float dvx_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

	if (gidz>=2+nPml && gidz<=nz-nPad-3-nPml && gidx>=2+nPml && gidx<=nx-3-nPml) {

	  dvz_dz = c1*(d_vz(gidz+1,gidx)-d_vz(gidz,gidx)) - c2*(d_vz(gidz+2,gidx)-d_vz(gidz-1,gidx));
	  dvx_dx = c1*(d_vx(gidz,gidx)-d_vx(gidz,gidx-1)) - c2*(d_vx(gidz,gidx+1)-d_vx(gidz,gidx-2));

	  d_szz(gidz,gidx) = d_szz(gidz,gidx) - \
	  	d_Lambda(gidz,gidx) * (dvz_dz/dz + dvx_dx/dx) * dt;

	}

	else {
		return;
	}
}
