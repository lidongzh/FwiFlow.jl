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
    return tran_lambda, tran_den
end

if !isdir("figures_summary")
  mkdir("figures_summary")
end

K = 20.0 .* ones(m,n) # millidarcy
K[8:10,:] .= 120.0
# K[17:21,:] .= 100.0
# for i = 1:m
#     for j = 1:n
#         if i <= (14 - 24)/(30 - 1)*(j-1) + 24 && i >= (12 - 18)/(30 - 1)*(j-1) + 18
#             K[i,j] = 100.0
#         end
#     end
# end
tfCtxTrue = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo, sw0, true)
out_sw_true, out_p_true = imseq(tfCtxTrue)
lambdas = Array{PyObject}(undef, n_survey)
dens = Array{PyObject}(undef, n_survey)
for i = 1:n_survey
    sw = out_sw_true[survey_indices[i]]
    p = out_p_true[survey_indices[i]]
    lambdas[i], dens[i] = sw_p_to_lambda_den(sw, p)
end

sess = Session();init(sess);

vps = Array{PyObject}(undef, n_survey)
for i=1:n_survey
  vps[i] = sqrt((lambdas[i] + 2.0 * tf_shear_sat1[i])/dens[i])
end
V = run(sess, vps);
S = run(sess, out_sw_true);

z_inj = (9-1)*h + h/2.0
x_inj = (3-1)*h + h/2.0
z_prod = (9-1)*h + h/2.0
x_prod = (28-1)*h + h/2.0

rc("axes", titlesize=30)
rc("axes", labelsize=30)
rc("xtick", labelsize=28)
rc("ytick", labelsize=28)
rc("legend", fontsize=30)
fig1,axs = subplots(3,3, figsize=[30,15], sharex=true, sharey=true)
ims = Array{Any}(undef, 9)
for iPrj = 1:3
  for jPrj = 1:3
    ims[(iPrj-1)*3+jPrj] = axs[iPrj,jPrj].imshow(V[(iPrj-1)*3+jPrj], extent=[0,n*h,m*h,0], vmin=3350, vmax=3500);
    axs[iPrj,jPrj].title.set_text("Stage $((iPrj-1)*3+jPrj)")
    if jPrj == 1 || jPrj == 1
      axs[iPrj,jPrj].set_ylabel("Depth (m)")
    end
    if iPrj == 3 || iPrj == 3
      axs[iPrj,jPrj].set_xlabel("Distance (m)")
    end
    # cb = fig1.colorbar(ims[(iPrj-1)*3+jPrj], ax=axs[iPrj,jPrj])
    # cb.set_label("Vp")
    axs[iPrj,jPrj].scatter(x_inj, z_inj, c="r", marker=">")
    axs[iPrj,jPrj].scatter(x_prod, z_prod, c="r", marker="<")
  end
end
fig1.subplots_adjust(wspace=0.02, hspace=0.18)
cbar_ax = fig1.add_axes([0.91, 0.08, 0.01, 0.82])
cb1 = fig1.colorbar(ims[1], cax=cbar_ax)
cb1.set_label("Vp (m/s)")
savefig("figures_summary/Vp_evo_patchy_true.pdf",bbox_inches="tight",pad_inches = 0);


fig2,axs = subplots(3,3, figsize=[30,15], sharex=true, sharey=true)
ims = Array{Any}(undef, 9)
for iPrj = 1:3
  for jPrj = 1:3
    ims[(iPrj-1)*3+jPrj] = axs[iPrj,jPrj].imshow(S[survey_indices[(iPrj-1)*3+jPrj], :, :], extent=[0,n*h,m*h,0], vmin=0.0, vmax=0.6);
    axs[iPrj,jPrj].title.set_text("Stage $((iPrj-1)*3+jPrj)")
    if jPrj == 1 || jPrj == 1
      axs[iPrj,jPrj].set_ylabel("Depth (m)")
    end
    if iPrj == 3 || iPrj == 3
      axs[iPrj,jPrj].set_xlabel("Distance (m)")
    end
    # if iPrj ==2 && jPrj == 3
    # cb = fig2.colorbar(ims[(iPrj-1)*3+jPrj], ax=axs[iPrj,jPrj])
    # cb.set_label("Saturation")
    axs[iPrj,jPrj].scatter(x_inj, z_inj, c="r", marker=">")
    axs[iPrj,jPrj].scatter(x_prod, z_prod, c="r", marker="<")
  end
end
# fig2.subplots_adjust(wspace=0.04, hspace=0.042)
fig2.subplots_adjust(wspace=0.02, hspace=0.18)
cbar_ax = fig2.add_axes([0.91, 0.08, 0.01, 0.82])
cb2 = fig2.colorbar(ims[1], cax=cbar_ax)
cb2.set_label("Saturation")
savefig("figures_summary/Saturation_evo_patchy_true.pdf",bbox_inches="tight",pad_inches = 0);


# iter = 100
# Prj_names = ["CO2", "CO2_1src", "CO2_2surveys", "CO2_6surveys"]
# K_name = "/K$iter.txt"


# fig,axs = subplots(2,2, figsize=[18,8], sharex=true, sharey=true)
# for iPrj = 1:2
#   for jPrj = 1:2
#     # println(ax)
#     A = readdlm(Prj_names[(iPrj-1)*2 + jPrj] * K_name)
#     im = axs[iPrj,jPrj].imshow(A, extent=[0,n*h,m*h,0]);
#     if jPrj == 1 || jPrj == 1
#       axs[iPrj,jPrj].set_ylabel("Depth (m)")
#     end
#     if iPrj == 2 || iPrj == 2
#       axs[iPrj,jPrj].set_xlabel("Distance (m)")
#     end
#     axs[iPrj,jPrj].text(-0.1,1.1,string("(" * Char((iPrj-1)*2 + jPrj+'a'-1) * ")"),transform=axs[iPrj,jPrj].transAxes,size=12,weight="bold")
#   end
# end
# fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
#                     wspace=0.1, hspace=0.2)
# cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
# cbar = fig.colorbar(im, cax=cb_ax)

# cb = fig.colorbar()
# clim([20, 120])
# cb.set_label("Permeability (md)")

# fig = figure()
# ax = fig.add_subplot(111)    # The big subplot
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

# # Turn off axis lines and ticks of the big subplot
# ax.spines["top"].set_color("none")
# ax.spines["bottom"].set_color("none")
# ax.spines["left"].set_color("none")
# ax.spines["right"].set_color("none")
# ax.tick_params(labelcolor="w", top="off", bottom="off", left="off", right="off")

# # Set common labels
# ax.set_xlabel("common xlabel")
# ax.set_ylabel("common ylabel")

# ax1.set_title('ax1 title')
# ax2.set_title('ax2 title')