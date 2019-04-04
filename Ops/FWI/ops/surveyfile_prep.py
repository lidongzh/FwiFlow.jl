import numpy as np
import json

def surveyfile_prep(z_src, x_src, z_rec, x_rec, survey_fname, Src_fname, idxPhase):
	x_src = x_src.tolist()
	z_src = z_src.tolist()
	x_rec = x_rec.tolist()
	z_rec = z_rec.tolist()
	nsrc = len(x_src)
	nrec = len(x_rec)
	survey = {}
	survey['nShots'] = nsrc
	# decision['same_num_rec_flag'] = 1;
	for i in range(0, nsrc):
	    shot = {}
	    shot['z_src'] = z_src[i]
	    shot['x_src'] = x_src[i]
	    shot['nrec'] = nrec
	    shot['z_rec'] = z_rec
	    shot['x_rec'] = x_rec
	    shot['Src_fname'] = Src_fname
	    survey['shot' + str(i)] = shot
	with open('../CUFD/Phase' + str(idxPhase) + '/Par/' + survey_fname, 'w') as fp:
	    json.dump(survey, fp)
