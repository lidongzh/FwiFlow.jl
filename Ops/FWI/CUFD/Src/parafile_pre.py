import numpy as np
import json

def parafile_pre(Cp, Cs, Den, nz, nx, dz, dx, nSteps, dt, f0, filter_para, nPml, isAc, withAdj, if_res, para_fname):

	nPad = 32 - (nz+2*nPml)%32
	# nPad = 0

	# Par_file_name = '../Par/Par_file.json'
	para = {}
	para['nz'] = nz + 2*nPml + nPad
	para['nx'] = nx + 2*nPml
	para['dz'] = dz
	para['dx'] = dx
	para['nSteps'] = nSteps
	para['dt'] = dt
	para['f0'] = f0
	para['nPoints_pml'] = nPml
	para['nPad'] = nPad
	para['Cp_fname'] = "../Models/Model_Cp.bin"
	para['Cs_fname'] = "../Models/Model_Cs.bin"
	para['Den_fname'] = "../Models/Model_Den.bin"
	para['data_dir_name'] = "../Data/"
	para['isAc'] = isAc
	para['withAdj'] = withAdj
	para['if_res'] = if_res
	para['if_win'] = False
	para['filter'] = filter_para
	para['if_src_update'] = True

	Cp_pad = np.pad(Cp, [(nPml, nPml+nPad),(nPml,nPml)], 'edge')
	Cs_pad = np.pad(Cs, [(nPml, nPml+nPad),(nPml,nPml)], 'edge')
	Den_pad = np.pad(Den, [(nPml, nPml+nPad),(nPml,nPml)], 'edge')

	Cp_pad.T.tofile(para['Cp_fname']);
	Cs_pad.T.tofile(para['Cs_fname']);
	Den_pad.T.tofile(para['Den_fname']);

	with open(para_fname, 'w') as fp:
		json.dump(para, fp)