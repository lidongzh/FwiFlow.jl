# input: nz, nx, dz, dx, nSteps, nPoints_pml, nPad, dt, f0, survey_fname, data_dir_name, scratch_dir_name, isAc
using JSON
using DataStructures

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

