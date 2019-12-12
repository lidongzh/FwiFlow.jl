# input: nz, nx, dz, dx, nSteps, nPoints_pml, nPad, dt, f0, survey_fname, data_dir_name, scratch_dir_name, isAc
using JSON
using DataStructures
using Dierckx

function paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, para_fname, survey_fname, data_dir_name; if_win=false, filter_para=nothing, if_src_update=false, scratch_dir_name::String="")

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
  if if_win != false
    para["if_win"] = true
  end
  if filter_para != nothing
    para["filter"] = filter_para
  end
  if if_src_update != false
    para["if_src_update"] = true
  end
	para["survey_fname"] = survey_fname
  para["data_dir_name"] = data_dir_name
  if !isdir(data_dir_name)
    mkdir(data_dir_name)
  end
  # if nStepsWrap != nothing
  #   para["nStepsWrap"] = nStepsWrap
  # end

	if(scratch_dir_name != "")
      para["scratch_dir_name"] = scratch_dir_name
      if !isdir(scratch_dir_name)
        mkdir(scratch_dir_name)
      end
  end
  para_string = JSON.json(para)

  open(para_fname,"w") do f
    write(f, para_string)
  end
end

# all shots share the same number of receivers
function surveyGen(z_src, x_src, z_rec, x_rec, survey_fname; Windows=nothing, Weights=nothing)
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
    if Windows != nothing
      shot["win_start"] = Windows["shot$(i-1)"][:start]
      shot["win_end"] = Windows["shot$(i-1)"][:end]
    end
    if Weights != nothing
      # shot["weights"] = Int64.(Weights["shot$(i-1)"][:weights])
      shot["weights"] = Weights["shot$(i-1)"][:weights]
    end
    survey["shot$(i-1)"] = shot
  end
  
  survey_string = JSON.json(survey)
  open(survey_fname,"w") do f
    write(f, survey_string)
  end

end

function sourceGene(f, nStep, delta_t)
#  Ricker wavelet generation and integration for source
#  Dongzhuo Li @ Stanford
#  May, 2015

  e = pi*pi*f*f;
  t_delay = 1.2/f;
  source = Matrix{Float64}(undef, 1, nStep)
  for it = 1:nStep
      source[it] = (1-2*e*(delta_t*(it-1)-t_delay)^2)*exp(-e*(delta_t*(it-1)-t_delay)^2);
  end

  for it = 2:nStep
      source[it] = source[it] + source[it-1];
  end
  source = source * delta_t;
end

# get vs high and low bounds from log point cloud 
# 1st row of Bounds: vp ref line
# 2nd row of Bounds: vs high ref line
# 3rd row of Bounds: vs low ref line
function cs_bounds_cloud(cpImg, Bounds)
  cs_high_itp = Spline1D(Bounds[1,:], Bounds[2,:]; k=1)
  cs_low_itp = Spline1D(Bounds[1,:], Bounds[3,:]; k=1)
  csHigh = zeros(size(cpImg))
  csLow = zeros(size(cpImg))
  for i = 1:size(cpImg, 1)
    for j = 1:size(cpImg, 2)
      csHigh[i,j] = cs_high_itp(cpImg[i,j])
      csLow[i,j] = cs_low_itp(cpImg[i,j])
    end
  end
  return csHigh, csLow
end

function klauderWave(fmin, fmax, t_sweep, nStepTotal, nStepDelay, delta_t)
#  Klauder wavelet
#  Dongzhuo Li @ Stanford
#  August, 2019
  nStep = nStepTotal - nStepDelay
  source = Matrix{Float64}(undef, 1, nStep+nStep-1)
  source_half = Matrix{Float64}(undef, 1, nStep-1)
  K = (fmax - fmin) / t_sweep
  f0 = (fmin + fmax) / 2.0
  t_axis = delta_t:delta_t:(nStep-1)*delta_t
  source_half = sin.(pi * K .* t_axis .* (t_sweep .- t_axis)) .* cos.(2.0 * pi * f0 .* t_axis) ./ (pi*K.*t_axis*t_sweep)
  for i = 1:nStep-1
    source[i] = source_half[end-i+1]
  end
  for i = nStep+1:2*nStep-1
    source[i] = source_half[i-nStep]
  end
  source[nStep] = 1.0
  source_crop = source[:,nStep-nStepDelay:end]
  return source_crop
end

# function klauderWave(fmin, fmax, t_sweep, nStep, delta_t)
# #  Klauder wavelet
# #  Dongzhuo Li @ Stanford
# #  August, 2019
#   source = Matrix{Float64}(undef, 1, nStep)
#   K = (fmax - fmin) / t_sweep
#   f0 = (fmin + fmax) / 2.0
#   t_axis = delta_t:delta_t:(nStep-1)*delta_t
#   source_part = sin.(pi * K .* t_axis .* (t_sweep .- t_axis)) .* cos.(2.0 * pi * f0 .* t_axis) ./ (pi*K.*t_axis*t_sweep)
#   for i = 2:nStep
#     source[i] = source_part[i-1]
#   end
#   source[1] = 1.0
#   return source
# end