# Generate Observed data from the true A-C equation

import numpy as np
import os
import matplotlib.pyplot as plt
import parafile_prep as pfp
import surveyfile_prep as sfp
import subprocess
import time

# Some parameters
nPhase = 4
nGpu = 1
# x_src = (np.arange(3, 384, 8)).astype(int)
# x_src = (np.arange(20,21)).astype(int)
# z_src = (np.ones(x_src.shape[0])).astype(int)
# x_rec = (np.arange(2,381)).astype(int)
# z_rec = (np.ones(x_rec.shape[0])).astype(int)

# x_src = (np.arange(10,60,10)).astype(int)
# z_src = (30*np.ones(x_src.shape[0])).astype(int)
# x_rec = (np.arange(21,71)).astype(int)
# z_rec = (30*np.ones(x_rec.shape[0])).astype(int)

x_src = (np.arange(5,300,10)).astype(int)
z_src = (1*np.ones(x_src.shape[0])).astype(int)
x_rec = (np.arange(5,300)).astype(int)
z_rec = (1*np.ones(x_rec.shape[0])).astype(int)

dz = 24
dx = 24
nSteps = 2001
dt = 0.0025
f0 = 4.5
filter_para = [0, 0.1, 100.0, 200.0]
nPml = 32
isAc = True

survey_fname = 'survey_file.json'

for idxPhase in range(1, nPhase):
	# mkdir
	os.system('mkdir ../CUFD/Phase%d' % idxPhase)
	os.system('mkdir ../CUFD/Phase%d/Bin' % idxPhase)
	os.system('mkdir ../CUFD/Phase%d/Data' % idxPhase)
	os.system('mkdir ../CUFD/Phase%d/Models' % idxPhase)
	os.system('mkdir ../CUFD/Phase%d/Par' % idxPhase)

	# parameter files
	Cp = np.loadtxt('../../AdvectionDiffusion/data/U' + str(idxPhase) + '.txt').astype('float32')
	(nz, nx) = Cp.shape
	print('nz = %d, nx = %d\n' % (nz, nx))
	Cs = np.zeros((nz, nx), dtype='float32')
	Den = 1000.0 * np.ones((nz, nx), dtype='float32')
	
	withAdj = False
	if_res = False
	para_fname = 'Par_file_obs_data.json'
	pfp.parafile_prep(Cp, Cs, Den, nz, nx, dz, dx, nSteps, dt, f0, filter_para, nPml, isAc, withAdj, if_res, para_fname, idxPhase)

	if_res = True
	para_fname = 'Par_file_calc_residual.json'
	pfp.parafile_prep(Cp, Cs, Den, nz, nx, dz, dx, nSteps, dt, f0, filter_para, nPml, isAc, withAdj, if_res, para_fname, idxPhase)
	withAdj = True
	para_fname = 'Par_file_calc_gradient.json'
	pfp.parafile_prep(Cp, Cs, Den, nz, nx, dz, dx, nSteps, dt, f0, filter_para, nPml, isAc, withAdj, if_res, para_fname, idxPhase)

	# survey files

	# Src_fname = '../../Mar_source_2001.bin'
	# Src_fname = '../../source_two1s.bin'
	Src_fname = '../../ricker_10Hz.bin'
	sfp.surveyfile_prep(z_src, x_src, z_rec, x_rec, survey_fname, Src_fname, idxPhase)

	os.system('cp ../CUFD/Src/CUFD ../CUFD/Phase' + str(idxPhase) + '/Bin/')

# forward simulations
for idxPhase in range(1, nPhase):
	gpu_id = idxPhase % nGpu
	gpu_usage = 'CUDA_VISIBLE_DEVICES=' + str(gpu_id)
	exe_cmd = ' ./CUFD ../Par/Par_file_obs_data.json ../Par/survey_file.json > out'
	cmd = gpu_usage + exe_cmd
	print(cmd)
	os.system('cd ../CUFD/Phase' + str(idxPhase) + '/Bin/;' + cmd)
	# pipes = subprocess.Popen('cd ../CUFD/Phase' + str(idxPhase) + '/Bin/;' + cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# while (pipes.poll() is None):
#     time.sleep(1)





