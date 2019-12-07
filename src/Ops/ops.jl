export laplacian_op, poisson_op, sat_op, upwlap_op, upwps_op, fwi_op, fwi_obs_op

OPS_DIR = @__DIR__

"""
"""
function fwi_op(args...)
    fwi_op = load_op_and_grad("$OPS_DIR/FWI/build/libFwiOp", "fwi_op")
    fwi_op(args...)
end


function fwi_obs_op(args...)
    fwi_obs_op = load_op("$OPS_DIR/FWI/build/libFwiOp", "fwi_obs_op")
    fwi_obs_op(args...)
end

function laplacian(args...)
    laplacian = load_op_and_grad("$OPS_DIR/Laplacian/build/libLaplacian", "laplacian")
    laplacian(args...)
end

function poisson_op(args...)
    poisson_op = load_op_and_grad("$OPS_DIR/Poisson/build/libPoissonOp", "poisson_op")
    poisson_op(args...)
end

function sat_op(args...)
    sat_op = load_op_and_grad("$OPS_DIR/Saturation/build/libSatOp", "sat_op")
    sat_op(args...)
end


function upwlap_op(args...)
    upwlap_op = load_op_and_grad("$OPS_DIR/Upwlap/build/libUpwlapOp", "upwlap_op")
    upwlap_op(args...)
end

function upwps_op(args...)
    upwps_op = load_op_and_grad("$OPS_DIR/Upwps/build/libUpwpsOp", "upwps_op")
    upwps_op(args...)
end