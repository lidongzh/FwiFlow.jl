# input: nz, nx, dz, dx, nSteps, nPoints_pml, nPad, dt, f0, survey_fname, data_dir_name, scratch_dir_name, isAc
using JSON
using DataStructures

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
      shot["weights"] = Int64.(Weights["shot$(i-1)"][:weights])
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



# __cnt = 0
# # invK = zeros(m,n)
# function print_loss(l, Lambda, Mu, Den, gradLambda, gradMu, gradDen, Stf)
#     global __cnt, __l, __Lambda, __gradLambda, __Mu, __gradMu, __Den, __gradDen, __Stf
#     if mod(__cnt,1)==0
#         println("\niter=$__iter, eval=$__cnt, current loss=",l)
#         # println("a=$a, b1=$b1, b2=$b2")
#     end
#     __cnt += 1
#     __l = l
#     __Lambda = Lambda
#     __gradLambda = gradLambda
#     __Mu = Mu
#     __gradMu = gradMu
#     __Den = Den
#     __gradDen = gradDen
#     __Stf = Stf
# end

# __iter = 0
# function print_iter(rk)
#     global __iter
#     if mod(__iter,1)==0
#         println("\n************* ITER=$__iter *************\n")
#     end
#     __iter += 1
#     open("./$(args["version"])/loss.txt", "a") do io 
#         writedlm(io, Any[__iter __l])
#     end
#     open("./$(args["version"])/Lambda$__iter.txt", "w") do io 
#         writedlm(io, __Lambda)
#     end
#     open("./$(args["version"])/gradLambda$__iter.txt", "w") do io
#          writedlm(io, __gradLambda)
#     end
#     open("./$(args["version"])/Mu$__iter.txt", "w") do io 
#         writedlm(io, __Mu)
#     end
#     open("./$(args["version"])/gradMu$__iter.txt", "w") do io
#          writedlm(io, __gradMu)
#     end
#     open("./$(args["version"])/Den$__iter.txt", "w") do io 
#         writedlm(io, __Den)
#     end
#     open("./$(args["version"])/gradDen$__iter.txt", "w") do io
#          writedlm(io, __gradDen)
#     end
#     open("./$(args["version"])/Stf$__iter.txt", "w") do io
#          writedlm(io, __Stf)
#     end
# end

