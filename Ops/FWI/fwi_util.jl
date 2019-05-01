# input: nz, nx, dz, dx, nSteps, nPoints_pml, nPad, dt, f0, survey_fname, data_dir_name, scratch_dir_name, isAc
using JSON
using DataStructures

if Sys.islinux()
py"""
import tensorflow as tf
libFwiOp = tf.load_op_library('build/libFwiOp.so')
@tf.custom_gradient
def fwi_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname):
    res = libFwiOp.fwi_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    def grad(dy):
        return libFwiOp.fwi_op_grad(dy, tf.constant(1.0,dtype=tf.float64),cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    return res, grad
def fwi_obs_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname):
    res = libFwiOp.fwi_obs_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    return res
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libFwiOp = tf.load_op_library('build/libFwiOp.dylib')
@tf.custom_gradient
def fwi_op(cp,cs,den):
    res = libFwiOp.fwi_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    def grad(dy):
        return libFwiOp.fwi_op_grad(dy,tf.constant(1.0,dtype=tf.float64),cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    return res, grad
def fwi_obs_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname):
    res = libFwiOp.fwi_obs_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    return res
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libFwiOp = tf.load_op_library('build/libFwiOp.dll')
@tf.custom_gradient
def fwi_op(cp,cs,den):
    res = libFwiOp.fwi_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    def grad(dy):
        return libFwiOp.fwi_op_grad(dy,tf.constant(1.0,dtype=tf.float64),cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    return res, grad
def fwi_obs_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname):
    res = libFwiOp.fwi_obs_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    return res
"""
end

fwi_op = py"fwi_op"
fwi_obs_op = py"fwi_obs_op"


function paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, filter_para, isAc, para_fname, survey_fname, data_dir_name, scratch_dir_name="")

  para = OrderedDict()
  para["nz"] = nz
	para["nx"] = nx
	para["dz"] = dz
	para["dx"] = dx
	para["nSteps"] = nSteps
	para["dt"] = dt
	para["f0"] = f0
	para["nPoints_pml"] = nPml
	para["nPad"] = nPad

	para["isAc"] = isAc
	para["if_win"] = false
	para["filter"] = filter_para
	para["if_src_update"] = false
	para["survey_fname"] = survey_fname
	para["data_dir_name"] = data_dir_name

	if(scratch_dir_name != "")
      para["scratch_dir_name"] = scratch_dir_name
  end
  para_string = JSON.json(para)

  open(para_fname,"w") do f
    write(f, para_string)
  end
end

# all shots share the same number of receivers
function surveyGen(z_src, x_src, z_rec, x_rec, survey_fname)
  nsrc = length(x_src)
  nrec = length(x_rec)
  survey = OrderedDict()
  survey["nShots"] = nsrc
  for i = 1:nsrc
    shot = OrderedDict()
    shot["z_src"] = z_src[i]
    shot["x_src"] = x_src[i]
    shot["nrec"] = nrec
    shot["z_rec"] = z_rec
    shot["x_rec"] = x_rec
    survey["shot$(i-1)"] = shot
  end
  
  survey_string = JSON.json(survey)
  open(survey_fname,"w") do f
    write(f, survey_string)
  end

end

