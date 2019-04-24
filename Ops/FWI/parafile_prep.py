import numpy as np
import json

def parafile_prep(Cp, Cs, Den, nz, nx, dz, dx, nSteps, dt, f0, filter_para, nPml, isAc, withAdj, if_res, para_fname, idxPhase):

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
	# para['Cp_fname'] = '../CUFD/Phase' + str(idxPhase) + '/Models/Model_Cp.bin'
	# para['Cs_fname'] = '../CUFD/Phase' + str(idxPhase) + '/Models/Model_Cs.bin'
	# para['Den_fname'] = '../CUFD/Phase' + str(idxPhase) + '/Models/Model_Den.bin'
	# para['data_dir_name'] = '../CUFD/Phase' + str(idxPhase) + '/Data/'
	para['Cp_fname'] = '../Models/Model_Cp.bin'
	para['Cs_fname'] = '../Models/Model_Cs.bin'
	para['Den_fname'] = '../Models/Model_Den.bin'
	para['data_dir_name'] = '../Data/'
	para['isAc'] = isAc
	para['withAdj'] = withAdj
	para['if_res'] = if_res
	para['if_win'] = False
	para['filter'] = filter_para
	para['if_src_update'] = False

	# Cp_pad = np.pad(Cp, [(nPml, nPml+nPad),(nPml,nPml)], 'edge')
	# Cs_pad = np.pad(Cs, [(nPml, nPml+nPad),(nPml,nPml)], 'edge')
	# Den_pad = np.pad(Den, [(nPml, nPml+nPad),(nPml,nPml)], 'edge')

	# Cp_pad.T.tofile('../CUFD/Phase' + str(idxPhase) + '/Models/Model_Cp.bin');
	# Cs_pad.T.tofile('../CUFD/Phase' + str(idxPhase) + '/Models/Model_Cs.bin');
	# Den_pad.T.tofile('../CUFD/Phase' + str(idxPhase) + '/Models/Model_Den.bin');
	Cp.T.tofile('../CUFD/Phase' + str(idxPhase) + '/Models/Model_Cp.bin');
	Cs.T.tofile('../CUFD/Phase' + str(idxPhase) + '/Models/Model_Cs.bin');
	Den.T.tofile('../CUFD/Phase' + str(idxPhase) + '/Models/Model_Den.bin');

	with open('../CUFD/Phase' + str(idxPhase) + '/Par/' + para_fname, 'w') as fp:
		json.dump(para, fp)