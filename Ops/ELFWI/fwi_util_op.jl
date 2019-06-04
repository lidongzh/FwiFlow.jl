
if Sys.islinux()
    py"""
    import tensorflow as tf
    libFwiOp = tf.load_op_library('./build/libFwiOp.so')
    @tf.custom_gradient
    def fwi_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname):
        misfit = libFwiOp.fwi_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname)
        def grad(dy):
            return libFwiOp.fwi_op_grad(dy, tf.constant(1.0,dtype=tf.float64),λ,μ,ρ,stf,gpu_id,shot_ids,para_fname)
        return misfit, grad
    def fwi_obs_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname):
        misfit = libFwiOp.fwi_obs_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname)
        return misfit
    """
    elseif Sys.isapple()
    py"""
    import tensorflow as tf
    libFwiOp = tf.load_op_library('./build/libFwiOp.dylib')
    @tf.custom_gradient
    def fwi_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname):
        misfit = libFwiOp.fwi_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname)
        def grad(dy):
            return libFwiOp.fwi_op_grad(dy, tf.constant(1.0,dtype=tf.float64),λ,μ,ρ,stf,gpu_id,shot_ids,para_fname)
        return misfit, grad
    def fwi_obs_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname):
        misfit = libFwiOp.fwi_obs_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname)
        return misfit
    """
    elseif Sys.iswindows()
    py"""
    import tensorflow as tf
    libFwiOp = tf.load_op_library('./build/libFwiOp.dll')
    @tf.custom_gradient
    def fwi_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname):
        misfit = libFwiOp.fwi_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname)
        def grad(dy):
            return libFwiOp.fwi_op_grad(dy, tf.constant(1.0,dtype=tf.float64),λ,μ,ρ,stf,gpu_id,shot_ids,para_fname)
        return misfit, grad
    def fwi_obs_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname):
        misfit = libFwiOp.fwi_obs_op(λ,μ,ρ,stf,gpu_id,shot_ids,para_fname)
        return misfit
    """
    end
    
    fwi_op = py"fwi_op"
    fwi_obs_op = py"fwi_obs_op"