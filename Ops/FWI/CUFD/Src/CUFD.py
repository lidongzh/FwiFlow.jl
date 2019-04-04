import numpy as np
import sys
import subprocess
import json
import time

def EL_CUFD(Cp, Cs, Den, nz, nx, dh, nSteps, dt, f0, nPml, decision_fname):

	start = time.time()
	nPad = 32 - nz%32
	# nPad = 0

	Par_file_name = '../Par/Par_file.json'
	para = {}
	para['nz'] = nz + 2*nPml + nPad
	para['nx'] = nx + 2*nPml
	para['dh'] = dh
	para['nSteps'] = nSteps
	para['dt'] = dt
	para['f0'] = f0
	para['nPoints_pml'] = nPml
	para['nPad'] = nPad
	para['Cp_fname'] = "../Models/Model_Cp.bin"
	para['Cs_fname'] = "../Models/Model_Cs.bin"
	para['Den_fname'] = "../Models/Model_Den.bin"
	para['data_dir_name'] = "../Data/"

	Cp_pad = np.pad(Cp, [(nPml, nPml+nPad),(nPml,nPml)], 'edge')
	Cs_pad = np.pad(Cs, [(nPml, nPml+nPad),(nPml,nPml)], 'edge')
	Den_pad = np.pad(Den, [(nPml, nPml+nPad),(nPml,nPml)], 'edge')

	Cp_pad.T.tofile(para['Cp_fname']);
	Cs_pad.T.tofile(para['Cs_fname']);
	Den_pad.T.tofile(para['Den_fname']);

	with open(Par_file_name, 'w') as fp:
		json.dump(para, fp)

	end = time.time()
	print("time = " + str(end - start))

	cmdlaunch='time CUDA_VISIBLE_DEVICES=7 ./CUFD ' + Par_file_name + ' ' + decision_fname + ' > out'
	print(cmdlaunch)
	pipes = subprocess.Popen(cmdlaunch,stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
	while (pipes.poll() is None):
		time.sleep(1)
	sys.stdout.write('Forward calculation completed \n')
	sys.stdout.flush()







