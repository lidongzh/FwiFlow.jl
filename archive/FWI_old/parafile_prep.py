import numpy as np
import json

def parafile_prep(nz, nx, dz, dx, nSteps, dt, f0, filter_para, nPml, isAc, para_fname, survey_fname, data_dir_name, scratch_dir_name=""):

	# nPad = 32 - (nz+2*nPml)%32
	nPad = 0

	# Par_file_name = '../Par/Par_file.json'
	para = {}
	# para['nz'] = nz + 2*nPml + nPad
	# para['nx'] = nx + 2*nPml
	para['nz'] = nz
	para['nx'] = nx
	para['dz'] = dz
	para['dx'] = dx
	para['nSteps'] = nSteps
	para['dt'] = dt
	para['f0'] = f0
	para['nPoints_pml'] = nPml
	para['nPad'] = nPad

	para['isAc'] = isAc
	para['if_win'] = False
	para['filter'] = filter_para
	para['if_src_update'] = False
	para['survey_fname'] = survey_fname
	para['data_dir_name'] = data_dir_name

	if(scratch_dir_name != ""):
  		para['scratch_dir_name'] = scratch_dir_name

	with open(para_fname, 'w') as fp:
		json.dump(para, fp)