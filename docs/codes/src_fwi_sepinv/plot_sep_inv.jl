include("args.jl")

function sw_p_to_lambda_den(sw, p)
    sw = tf.reshape(sw, (1, m, n, 1))
    p = tf.reshape(p, (1, m, n, 1))
    sw = tf.image.resize_bilinear(sw, (nz, nx))
    p = tf.image.resize_bilinear(p, (nz, nx))
    sw = cast(sw, Float64)
    p = cast(p, Float64)
    sw = squeeze(sw)
    p = squeeze(p)
    # tran_lambda, tran_den = Gassman(sw)
    # tran_lambda, tran_den = RockLinear(sw) # test linear relationship
    tran_lambda, tran_den = Patchy(sw)
    # tran_lambda_pad =  tf.pad(tran_lambda, [nPml (nPml+nPad); nPml nPml], constant_values=3500.0^2*2200.0/3.0) /1e6
    # tran_den_pad = tf.pad(tran_den, [nPml (nPml+nPad); nPml nPml], constant_values=2200.0)
    return tran_lambda, tran_den
end

if !isdir("figures_summary")
  mkdir("figures_summary")
end

lambdasObs = Array{PyObject}(undef, n_survey-1)
densObs = Array{PyObject}(undef, n_survey-1)
for iSur = 2:n_survey
  lp = readdlm("./CO2/FWI_stage$(iSur)/loss.txt")
  Lp = Int64((lp[end,1]))
  lambdasObs[iSur-1] = readdlm("./CO2/FWI_stage$(iSur)/Lambda$Lp.txt")
  densObs[iSur-1] = readdlm("./CO2/FWI_stage$(iSur)/Den$Lp.txt")
end

m = 15
n = 30
h = 30.0
dz = 3.0 # meters
dx = 3.0
nz = Int64(round((m * h) / dz)) + 1
nx = Int64(round((n * h) / dx)) + 1
z_src = (collect(5:10:nz-5) .- 1 ) .* dz .+ dz/2.0
x_src = (5-1)ones(Int64, size(z_src)) .* dx .+ dx/2.0
z_rec = (collect(5:1:nz-5) .- 1) .* dz .+ dz/2.0
x_rec = (nx-5-1) .* ones(Int64, size(z_rec)) .*dx .+ dx/2.0
z_inj = (9-1)*h + h/2.0
x_inj = (3-1)*h + h/2.0
z_prod = (9-1)*h + h/2.0
x_prod = (28-1)*h + h/2.0

rc("axes", titlesize=20)
rc("axes", labelsize=18)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("legend", fontsize=20)
figure()
K = readdlm("CO2/flow_fit_results/K100.txt")
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
scatter(x_inj, z_inj, c="r", marker=">")
scatter(x_prod, z_prod, c="r", marker="<")
savefig("figures_summary/K_sep_fit.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

rc("axes", titlesize=30)
rc("axes", labelsize=30)
rc("xtick", labelsize=28)
rc("ytick", labelsize=28)
rc("legend", fontsize=30)
fig1,axs = subplots(3,3, figsize=[30,15], sharex=true, sharey=true)
ims = Array{Any}(undef, 9)
for iPrj = 1:3
  for jPrj = 1:3
    ims[(iPrj-1)*3+jPrj] = axs[iPrj,jPrj].imshow(lambdasObs[(iPrj-1)*3+jPrj], extent=[0,n*h,m*h,0], vmin=5500, vmax=9000);
    axs[iPrj,jPrj].title.set_text("Snapshot $((iPrj-1)*3+jPrj+1)")
    if jPrj == 1 || jPrj == 1
      axs[iPrj,jPrj].set_ylabel("Depth (m)")
    end
    if iPrj == 3 || iPrj == 3
      axs[iPrj,jPrj].set_xlabel("Distance (m)")
    end
    # cb = fig1.colorbar(ims[(iPrj-1)*3+jPrj], ax=axs[iPrj,jPrj])
    # cb.set_label("λ")
    axs[iPrj,jPrj].scatter(x_inj, z_inj, c="r", marker=">", s=128)
    axs[iPrj,jPrj].scatter(x_prod, z_prod, c="r", marker="<", s=128)
  end
end
# fig1.subplots_adjust(wspace=0.02, hspace=0.042)
fig1.subplots_adjust(wspace=0.02, hspace=0.18)
cbar_ax = fig1.add_axes([0.91, 0.08, 0.01, 0.82])
cb1 = fig1.colorbar(ims[1], cax=cbar_ax)
cb1.set_label("λ (MPa)")
savefig("figures_summary/Lambda_FWI_sep_inv.pdf",bbox_inches="tight",pad_inches = 0);