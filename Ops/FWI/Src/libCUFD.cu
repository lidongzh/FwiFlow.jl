// Dongzhuo Li 05/06/2018
#include <chrono>
#include <string>
#include "Boundary.h"
#include "Cpml.h"
#include "Model.h"
#include "Parameter.h"
#include "Src_Rec.h"
#include "utilities.h"
using std::string;

#define VERBOSE
#define DEBUG

// extern "C" void cufd(double *res, double *grad_Cp, double *grad_Cs,
//                      double *grad_Den, double *grad_stf, const double *Cp,
//                      const double *Cs, const double *Den, const double *stf,
//                      int calc_id, const int gpu_id, int group_size,
//                      const int *shot_ids, const string para_fname);

/*
        double res : residual
        double *grad_Cp : gradients of Cp (p-wave velocity)
        double *grad_Cs : gradients of Cs (s-wave velocity)
        double *grad_Den : gradients of density
        double *grad_stf : gradients of source time function
        double *Cp : p-wave velocity
        double *Cs : s-wave velocity
        double *Den : density
        double *stf : source time function of all shots
        int calc_id :
                                        calc_id = 0  -- compute residual
                                        calc_id = 1  -- compute gradient
                                        calc_id = 2  -- compute observation only
        int gpu_id  :   CUDA_VISIBLE_DEVICES
        int group_size: number of shots in the group
        int *shot_ids :   processing shot shot_ids
        string para_fname :  parameter path
        // string survey_fname :  survey file (src/rec) path
        // string data_dir : data directory
        // string scratch_dir : temporary files
*/
void cufd(double *res, double *grad_Cp, double *grad_Cs, double *grad_Den,
          double *grad_stf, const double *Cp, const double *Cs,
          const double *Den, const double *stf, int calc_id, const int gpu_id,
          int group_size, const int *shot_ids, const string para_fname) {
  // int deviceCount = 0;
  // CHECK(cudaGetDeviceCount (&deviceCount));
  // printf("number of devices = %d\n", deviceCount);
  CHECK(cudaSetDevice(gpu_id));
  auto start0 = std::chrono::high_resolution_clock::now();

  // std::string para_fname = para_dir + "/fwi_param.json";
  // std::string survey_fname = "/survey_file.json";
  if (calc_id < 0 || calc_id > 2) {
    printf("Invalid calc_id %d\n", calc_id);
    exit(0);
  }

  // NOTE Read parameter file
  Parameter para(para_fname, calc_id);
  int nz = para.nz();
  int nx = para.nx();
  int nPml = para.nPoints_pml();
  int nPad = para.nPad();
  float dz = para.dz();
  float dx = para.dx();
  float dt = para.dt();
  float f0 = para.f0();

  int iSnap = 0;  // 400
  int nrec = 1;
  float win_ratio = 0.005;
  int nSteps = para.nSteps();
  float amp_ratio = 1.0;

  // transpose models and convert to float
  float *fCp, *fCs, *fDen;
  fCp = (float *)malloc(nz * nx * sizeof(float));
  fCs = (float *)malloc(nz * nx * sizeof(float));
  fDen = (float *)malloc(nz * nx * sizeof(float));
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      fCp[j * nz + i] = Cp[i * nx + j];
      fCs[j * nz + i] = Cs[i * nx + j];
      fDen[j * nz + i] = Den[i * nx + j];
    }
  }
  Model model(para, fCp, fCs, fDen);
  // Model model;
  Cpml cpml(para, model);
  Bnd boundaries(para);

  auto startSrc = std::chrono::high_resolution_clock::now();
  Src_Rec src_rec(para, para.survey_fname(), stf, group_size, shot_ids);
  // TODO: group_size -> shot group size
  auto finishSrc = std::chrono::high_resolution_clock::now();
#ifdef VERBOSE
  std::chrono::duration<double> elapsedSrc = finishSrc - startSrc;
  std::cout << "Src_Rec time: " << elapsedSrc.count() << " second(s)"
            << std::endl;
  std::cout << "number of shots " << src_rec.d_vec_z_rec.size() << std::endl;
  std::cout << "number of d_data " << src_rec.d_vec_data.size() << std::endl;
#endif

  // compute Courant number
  compCourantNumber(model.h_Cp, nz * nx, dt, dz, dx);

  dim3 threads(TX, TY);
  dim3 blocks((nz + TX - 1) / TX, (nx + TY - 1) / TY);
  dim3 threads2(TX + 4, TY + 4);
  dim3 blocks2((nz + TX + 3) / (TX + 4), (nx + TY + 3) / (TY + 4));

  float *d_vz, *d_vx, *d_szz, *d_sxx, *d_sxz, *d_vz_adj, *d_vx_adj, *d_szz_adj,
      *d_szz_p1;
  float *d_mem_dvz_dz, *d_mem_dvz_dx, *d_mem_dvx_dz, *d_mem_dvx_dx;
  float *d_mem_dszz_dz, *d_mem_dsxx_dx, *d_mem_dsxz_dz, *d_mem_dsxz_dx;
  float *d_mat_dvz_dz, *d_mat_dvx_dx;
  float *d_l2Obj_temp;
  float *h_l2Obj_temp = NULL;
  h_l2Obj_temp = (float *)malloc(sizeof(float));
  float h_l2Obj = 0.0;
  CHECK(cudaMalloc((void **)&d_vz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_szz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vz_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vx_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_szz_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_szz_p1, nz * nx * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_mem_dvz_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvz_dx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvx_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvx_dx, nz * nx * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_mem_dszz_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxx_dx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxz_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxz_dx, nz * nx * sizeof(float)));
  // spatial derivatives: for kernel computations
  CHECK(cudaMalloc((void **)&d_mat_dvz_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mat_dvx_dx, nz * nx * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_l2Obj_temp, 1 * sizeof(float)));

  float *h_snap, *h_snap_back, *h_snap_adj;
  h_snap = (float *)malloc(nz * nx * sizeof(float));
  h_snap_back = (float *)malloc(nz * nx * sizeof(float));
  h_snap_adj = (float *)malloc(nz * nx * sizeof(float));

  cudaStream_t streams[group_size];

  auto finish0 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed0 = finish0 - start0;
#ifdef VERBOSE
  std::cout << "Initialization time: " << elapsed0.count() << " second(s)"
            << std::endl;
#endif

  auto start = std::chrono::high_resolution_clock::now();

  // NOTE Processing Shot
  for (int iShot = 0; iShot < group_size; iShot++) {
#ifdef VERBOSE
    printf("	Processing shot %d\n", shot_ids[iShot]);
#endif
    CHECK(cudaStreamCreate(&streams[iShot]));

    // load precomputed presure DL
    // fileBinLoad(h_snap, nz*nx, "Pressure.bin");
    // CHECK(cudaMemcpy(d_szz, h_snap, nz*nx*sizeof(float),
    // cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_vx, h_snap,
    // nz*nx*sizeof(float), cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_vz,
    // h_snap, nz*nx*sizeof(float), cudaMemcpyHostToDevice));

    intialArrayGPU<<<blocks, threads>>>(d_vz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_vx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_szz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_sxx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_sxz, nz, nx, 0.0);

    intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dx, nz, nx, 0.0);

    intialArrayGPU<<<blocks, threads>>>(d_mem_dszz_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dsxx_dx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dx, nz, nx, 0.0);

    intialArrayGPU<<<blocks, threads>>>(d_mat_dvz_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mat_dvx_dx, nz, nx, 0.0);

    nrec = src_rec.vec_nrec.at(iShot);
    if (para.if_res()) {
      fileBinLoad(src_rec.vec_data_obs.at(iShot), nSteps * nrec,
                  para.data_dir_name() + "/Shot" +
                      std::to_string(shot_ids[iShot]) + ".bin");
      CHECK(cudaMemcpyAsync(src_rec.d_vec_data_obs.at(iShot),
                            src_rec.vec_data_obs.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));
    }
    // ------------------------------------ time loop
    // ------------------------------------
    for (int it = 0; it <= nSteps - 2; it++) {
      // =========================== elastic or acoustic
      // ===========================
      if (para.withAdj()) {
        // save and record from the beginning
        boundaries.field_from_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it);
      }

      // get snapshot at time it
      if (it == iSnap && iShot == 0) {
        CHECK(cudaMemcpy(h_snap, d_szz, nz * nx * sizeof(float),
                         cudaMemcpyDeviceToHost));
      }

      if (para.isAc()) {
        ac_pressure<<<blocks, threads>>>(
            d_vz, d_vx, d_szz, d_mem_dvz_dz, d_mem_dvx_dx, model.d_Lambda,
            model.d_Den, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half,
            cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, nz, nx, dt, dz, dx, nPml, nPad,
            true, d_mat_dvz_dz, d_mat_dvx_dx);

        add_source<<<1, 1>>>(d_szz, d_sxx, src_rec.vec_source.at(iShot)[it], nz,
                             true, src_rec.vec_z_src.at(iShot),
                             src_rec.vec_x_src.at(iShot), dt, model.d_Cp);

        ac_velocity<<<blocks, threads>>>(
            d_vz, d_vx, d_szz, d_mem_dszz_dz, d_mem_dsxx_dx, model.d_Lambda,
            model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b, cpml.d_K_z,
            cpml.d_a_z, cpml.d_b_z, cpml.d_K_x_half, cpml.d_a_x_half,
            cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, true);
      } else {
        el_stress<<<blocks, threads>>>(
            d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dvz_dz, d_mem_dvz_dx,
            d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu,
            model.d_ave_Mu, model.d_Den, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
            cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x,
            cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half,
            cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, true);

        add_source<<<1, 1>>>(d_szz, d_sxx, src_rec.vec_source.at(iShot)[it], nz,
                             true, src_rec.vec_z_src.at(iShot),
                             src_rec.vec_x_src.at(iShot), dt, model.d_Cp);

        el_velocity<<<blocks, threads>>>(
            d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dszz_dz, d_mem_dsxz_dx,
            d_mem_dsxz_dz, d_mem_dsxx_dx, model.d_Lambda, model.d_Mu,
            model.d_ave_Byc_a, model.d_ave_Byc_b, cpml.d_K_z, cpml.d_a_z,
            cpml.d_b_z, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half,
            cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half,
            cpml.d_a_x_half, cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad,
            true);
      }
      recording<<<(nrec + 31) / 32, 32>>>(
          d_szz, nz, src_rec.d_vec_data.at(iShot), iShot, it + 1, nSteps, nrec,
          src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot));
    }

    if (!para.if_res()) {
      CHECK(cudaMemcpyAsync(
          src_rec.vec_data.at(iShot), src_rec.d_vec_data.at(iShot),
          nSteps * nrec * sizeof(float), cudaMemcpyDeviceToHost,
          streams[iShot]));  // test
    }

    // fileBinWrite(h_snap, nz*nx, "SnapGPU.bin");

    // compute residuals
    if (para.if_res()) {
      dim3 blocksT((nSteps + TX - 1) / TX, (nrec + TY - 1) / TY);

      // for fun modify observed data
      // float filter2[4] = {8.0, 9.0, 12.0, 13.0};
      // cuda_window<<<blocksT,threads>>>(nSteps, nrec, dt, win_ratio,
      // src_rec.d_vec_data_obs.at(iShot)); bp_filter1d(nSteps, dt, nrec,
      // src_rec.d_vec_data_obs.at(iShot), filter2);

      // windowing
      if (para.if_win()) {
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            win_ratio, src_rec.d_vec_data_obs.at(iShot));
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            win_ratio, src_rec.d_vec_data.at(iShot));
      } else {
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio,
                                          src_rec.d_vec_data_obs.at(iShot));
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio,
                                          src_rec.d_vec_data.at(iShot));
      }

      // filtering
      if (para.if_filter()) {
        bp_filter1d(nSteps, dt, nrec, src_rec.d_vec_data_obs.at(iShot),
                    para.filter());
        bp_filter1d(nSteps, dt, nrec, src_rec.d_vec_data.at(iShot),
                    para.filter());
      }

      // Calculate source update and filter calculated data
      if (para.if_src_update()) {
        amp_ratio =
            source_update(nSteps, dt, nrec, src_rec.d_vec_data_obs.at(iShot),
                          src_rec.d_vec_data.at(iShot),
                          src_rec.d_vec_source.at(iShot), src_rec.d_coef);
        printf("	Source update => Processing shot %d, amp_ratio = %f\n",
               iShot, amp_ratio);
      }
      amp_ratio = 1.0;  // amplitude not used, so set to 1.0

      // objective function
      gpuMinus<<<blocksT, threads>>>(
          src_rec.d_vec_res.at(iShot), src_rec.d_vec_data_obs.at(iShot),
          src_rec.d_vec_data.at(iShot), nSteps, nrec);
      cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp, src_rec.d_vec_res.at(iShot),
                                     nSteps * nrec);
      CHECK(cudaMemcpy(h_l2Obj_temp, d_l2Obj_temp, sizeof(float),
                       cudaMemcpyDeviceToHost));
      h_l2Obj += h_l2Obj_temp[0];

      //  update source again (adjoint)
      if (para.if_src_update()) {
        source_update_adj(nSteps, dt, nrec, src_rec.d_vec_res.at(iShot),
                          amp_ratio, src_rec.d_coef);
      }

      // filtering again (adjoint)
      if (para.if_filter()) {
        bp_filter1d(nSteps, dt, nrec, src_rec.d_vec_res.at(iShot),
                    para.filter());
      }
      // windowing again (adjoint)
      if (para.if_win()) {
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            0.1, src_rec.d_vec_res.at(iShot));
      } else {
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio,
                                          src_rec.d_vec_res.at(iShot));
      }

      CHECK(cudaMemcpyAsync(
          src_rec.vec_res.at(iShot), src_rec.d_vec_res.at(iShot),
          nSteps * nrec * sizeof(float), cudaMemcpyDeviceToHost,
          streams[iShot]));  // test
      // CHECK(cudaMemcpy(src_rec.vec_res.at(iShot), src_rec.d_vec_res.at(iShot), \
			// 	nSteps*nrec*sizeof(float), cudaMemcpyDeviceToHost)); // test
      CHECK(cudaMemcpyAsync(
          src_rec.vec_data.at(iShot), src_rec.d_vec_data.at(iShot),
          nSteps * nrec * sizeof(float), cudaMemcpyDeviceToHost,
          streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(
          src_rec.vec_data_obs.at(iShot), src_rec.d_vec_data_obs.at(iShot),
          nSteps * nrec * sizeof(float), cudaMemcpyDeviceToHost,
          streams[iShot]));  // save preconditioned observed
      CHECK(cudaMemcpy(src_rec.vec_source.at(iShot),
                       src_rec.d_vec_source.at(iShot), nSteps * sizeof(float),
                       cudaMemcpyDeviceToHost));
    }
    // =================
    cudaDeviceSynchronize();

    if (para.withAdj()) {
      // ------------------------------------- Backward
      // ---------------------------------- initialization
      intialArrayGPU<<<blocks, threads>>>(d_vz_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_vx_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_szz_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_szz_p1, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dz, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dx, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dszz_dz, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dsxx_dx, nz, nx, 0.0);

      for (int it = nSteps - 2; it >= 0; it--) {
        if (para.isAc()) {
          // if (it <= nSteps - 2) {
          // save p to szz_plus_one
          assignArrayGPU<<<blocks, threads>>>(d_szz, d_szz_p1, nz, nx);
          // value at T-1
          ac_velocity<<<blocks, threads>>>(
              d_vz, d_vx, d_szz, d_mem_dszz_dz, d_mem_dsxx_dx, model.d_Lambda,
              model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b, cpml.d_K_z,
              cpml.d_a_z, cpml.d_b_z, cpml.d_K_x_half, cpml.d_a_x_half,
              cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, false);
          boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it, false);

          add_source<<<1, 1>>>(d_szz, d_sxx, src_rec.vec_source.at(iShot)[it],
                               nz, false, src_rec.vec_z_src.at(iShot),
                               src_rec.vec_x_src.at(iShot), dt, model.d_Cp);
          add_source<<<1, 1>>>(d_szz_p1, d_sxx,
                               src_rec.vec_source.at(iShot)[it], nz, false,
                               src_rec.vec_z_src.at(iShot),
                               src_rec.vec_x_src.at(iShot), dt, model.d_Cp);

          ac_pressure<<<blocks, threads>>>(
              d_vz, d_vx, d_szz, d_mem_dvz_dz, d_mem_dvx_dx, model.d_Lambda,
              model.d_Den, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half,
              cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, nz, nx, dt, dz, dx, nPml,
              nPad, false, d_mat_dvz_dz, d_mat_dvx_dx);

          boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it, true);
          // value at T-2

          // ================
          // adjoint computation

          ac_velocity_adj<<<blocks, threads>>>(
              d_vz_adj, d_vx_adj, d_szz_adj, d_mem_dvz_dz, d_mem_dvx_dx,
              d_mem_dszz_dz, d_mem_dsxx_dx, model.d_Lambda, model.d_Den,
              model.d_ave_Byc_a, model.d_ave_Byc_b, cpml.d_K_z_half,
              cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x_half,
              cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_z, cpml.d_a_z,
              cpml.d_b_z, cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, nz, nx, dt, dz,
              dx, nPml, nPad);

          // inject residuals
          res_injection<<<(nrec + 31) / 32, 32>>>(
              d_szz_adj, nz, src_rec.d_vec_res.at(iShot), model.d_Lambda,
              it + 1, dt, nSteps, nrec, src_rec.d_vec_z_rec.at(iShot),
              src_rec.d_vec_x_rec.at(iShot));

          ac_pressure_adj<<<blocks, threads>>>(
              d_vz_adj, d_vx_adj, d_szz_adj, d_mem_dvz_dz, d_mem_dvx_dx,
              d_mem_dszz_dz, d_mem_dsxx_dx, model.d_Lambda, model.d_Den,
              model.d_ave_Byc_a, model.d_ave_Byc_b, cpml.d_K_z_half,
              cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x_half,
              cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_z, cpml.d_a_z,
              cpml.d_b_z, cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, nz, nx, dt, dz,
              dx, nPml, nPad, model.d_Cp, d_mat_dvz_dz, d_mat_dvx_dx,
              model.d_CpGrad);
          // value at T-1

          // ac_adj_push<<<blocks,threads2>>>(d_vz_adj, d_vx_adj, d_szz_adj, d_adj_temp, \
					// 		d_mem_dvz_dz, d_mem_dvx_dx, d_mem_dszz_dz, d_mem_dsxx_dx, \
					// 		model.d_Lambda, model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b, \
					// 		cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, \
					// 		cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, \
					// 		cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, \
					// 		cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, \
					// 		nz, nx, dt, dz, dx, nPml, nPad);

          // image_vel<<<blocks,threads>>>(d_szz_adj, nz, nx, dt, dz, dx, nPml, nPad, \
     			//         model.d_Cp, model.d_Den, d_mat_dvz_dz, d_mat_dvx_dx, model.d_CpGrad);
          image_vel_time<<<blocks, threads>>>(
              d_szz, d_szz_p1, d_szz_adj, nz, nx, dt, dz, dx, nPml, nPad,
              model.d_Cp, model.d_Lambda, model.d_CpGrad);

        } else {
          el_velocity<<<blocks, threads>>>(
              d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dszz_dz, d_mem_dsxz_dx,
              d_mem_dsxz_dz, d_mem_dsxx_dx, model.d_Lambda, model.d_Mu,
              model.d_ave_Byc_a, model.d_ave_Byc_b, cpml.d_K_z, cpml.d_a_z,
              cpml.d_b_z, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half,
              cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half,
              cpml.d_a_x_half, cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad,
              false);

          el_stress<<<blocks, threads>>>(
              d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dvz_dz, d_mem_dvz_dx,
              d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu,
              model.d_ave_Mu, model.d_Den, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
              cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x,
              cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half,
              cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, false);
        }

        // boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it);

        if (it == iSnap && iShot == 0) {
          CHECK(cudaMemcpy(h_snap_back, d_szz, nz * nx * sizeof(float),
                           cudaMemcpyDeviceToHost));
          CHECK(cudaMemcpy(h_snap_adj, d_szz_adj, nz * nx * sizeof(float),
                           cudaMemcpyDeviceToHost));
        }
        if (iShot == 0) {
          // CHECK(cudaMemcpy(h_snap_adj, d_szz_adj, nz*nx*sizeof(float),
          // cudaMemcpyDeviceToHost)); fileBinWrite(h_snap_adj, nz*nx,
          // "SnapGPU_adj_" + std::to_string(it) + ".bin");
          // CHECK(cudaMemcpy(h_snap, d_szz, nz*nx*sizeof(float),
          // cudaMemcpyDeviceToHost)); fileBinWrite(h_snap, nz*nx, "SnapGPU_"
          // + std::to_string(it) + ".bin");
        }
      }
      // fileBinWrite(h_snap_back, nz*nx, "SnapGPU_back.bin");
      // fileBinWrite(h_snap_adj, nz*nx, "SnapGPU_adj.bin");
      CHECK(cudaMemcpy(model.h_CpGrad, model.d_CpGrad, nz * nx * sizeof(float),
                       cudaMemcpyDeviceToHost));
      // fileBinWrite(model.h_CpGrad, nz*nx, "CpGradient.bin");

      for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
          grad_Cp[i * nx + j] = model.h_CpGrad[j * nz + i];
        }
      }
      initialArray(grad_Cs, nz * nx, 0.0);
      initialArray(grad_Den, nz * nx, 0.0);
      initialArray(grad_stf, nSteps * src_rec.nShots, 0.0);
    }
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
#ifdef VERBOSE
  std::cout << "Elapsed time: " << elapsed.count() << " second(s)."
            << std::endl;
#endif

  if (!para.if_res()) {
    for (int iShot = 0; iShot < group_size; iShot++) {
      fileBinWrite(src_rec.vec_data.at(iShot),
                   nSteps * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot" +
                       std::to_string(shot_ids[iShot]) + ".bin");
    }
  }

  if (para.if_save_scratch()) {
    for (int iShot = 0; iShot < group_size; iShot++) {
      fileBinWrite(src_rec.vec_res.at(iShot),
                   nSteps * src_rec.vec_nrec.at(iShot),
                   para.scratch_dir_name() + "/Residual_Shot" +
                       std::to_string(shot_ids[iShot]) + ".bin");
      fileBinWrite(src_rec.vec_data.at(iShot),
                   nSteps * src_rec.vec_nrec.at(iShot),
                   para.scratch_dir_name() + "/Syn_Shot" +
                       std::to_string(shot_ids[iShot]) + ".bin");
      fileBinWrite(src_rec.vec_data_obs.at(iShot),
                   nSteps * src_rec.vec_nrec.at(iShot),
                   para.scratch_dir_name() + "/CondObs_Shot" +
                       std::to_string(shot_ids[iShot]) + ".bin");
      // fileBinWrite(src_rec.vec_source.at(iShot), nSteps,
      //              para.scratch_dir_name() + "src_updated" +
      //                  std::to_string(iShot) + ".bin");
    }
  }
  // #ifdef DEBUG
  //   std::cout << "cufd--" << __LINE__ << std::endl;
  // #endif

  // output residual
  if (para.if_res() && !para.withAdj()) {
    h_l2Obj = 0.5 * h_l2Obj;  // DL 02/21/2019 (need to make misfit accurate
                              // here rather than in the script)
    // fileBinWrite(&h_l2Obj, 1, "l2Obj.bin");
    std::cout << "Total l2 residual = " << std::to_string(h_l2Obj) << std::endl;
    std::cout << "calc_id = " << calc_id << std::endl;
    *res = h_l2Obj;
  }

  free(h_l2Obj_temp);

  free(h_snap);

  free(h_snap_back);

  free(h_snap_adj);

  free(fCp);

  free(fCs);

  free(fDen);

  // destroy the streams
  for (int iShot = 0; iShot < group_size; iShot++)
    CHECK(cudaStreamDestroy(streams[iShot]));

  cudaFree(d_vz);
  cudaFree(d_vx);
  cudaFree(d_szz);
  cudaFree(d_sxx);
  cudaFree(d_sxz);
  cudaFree(d_vz_adj);
  cudaFree(d_vx_adj);
  cudaFree(d_szz_adj);
  cudaFree(d_szz_p1);
  cudaFree(d_mem_dvz_dz);
  cudaFree(d_mem_dvz_dx);
  cudaFree(d_mem_dvx_dz);
  cudaFree(d_mem_dvx_dx);
  cudaFree(d_mem_dszz_dz);
  cudaFree(d_mem_dsxx_dx);
  cudaFree(d_mem_dsxz_dz);
  cudaFree(d_mem_dsxz_dx);
  cudaFree(d_mat_dvz_dz);
  cudaFree(d_mat_dvx_dx);
  cudaFree(d_l2Obj_temp);

#ifdef VERBOSE
  std::cout << "Done!" << std::endl;
#endif
}
