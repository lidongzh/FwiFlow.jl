using PyPlot
using DelimitedFiles

if !isdir("figures_summary")
  mkdir("figures_summary")
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

iter = 100
Prj_names = ["CO2", "CO2_1src", "CO2_2surveys", "Brie_3_nocoefupdate", "Brie_tune_coef_true3_start2", "Brie_true3_set2_noupdate", "Brie_tune_coef_true3_start2_scale30"]
K_name = "/K$iter.txt"

rc("axes", titlesize=20)
rc("axes", labelsize=18)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("legend", fontsize=20)
# true model
figure()
K = 20.0 .* ones(m,n)
K[8:10,:] .= 120.0
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
scatter(x_inj, z_inj, c="r", marker=">", s=64)
scatter(x_prod, z_prod, c="r", marker="<", s=64)
savefig("figures_summary/K_true.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

# init model
figure()
K = 20.0 .* ones(m,n)
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
scatter(x_inj, z_inj, c="r", marker=">", s=64)
scatter(x_prod, z_prod, c="r", marker="<", s=64)
savefig("figures_summary/K_init.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

figure()
iPrj = 1
K = readdlm(Prj_names[iPrj] * K_name)
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
scatter(x_inj, z_inj, c="r", marker=">", s=64)
scatter(x_prod, z_prod, c="r", marker="<", s=64)
savefig("figures_summary/K_$(Prj_names[iPrj]).pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

figure()
iPrj = 3
K = readdlm(Prj_names[iPrj] * K_name)
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
scatter(x_inj, z_inj, c="r", marker=">", s=64)
scatter(x_prod, z_prod, c="r", marker="<", s=64)
savefig("figures_summary/K_$(Prj_names[iPrj]).pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

figure()
iPrj = 4
K = readdlm(Prj_names[iPrj] * K_name)
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
scatter(x_inj, z_inj, c="r", marker=">", s=64)
scatter(x_prod, z_prod, c="r", marker="<", s=64)
savefig("figures_summary/K_$(Prj_names[iPrj]).pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

figure()
iPrj = 5
K = readdlm(Prj_names[iPrj] * K_name)
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
scatter(x_inj, z_inj, c="r", marker=">", s=64)
scatter(x_prod, z_prod, c="r", marker="<", s=64)
savefig("figures_summary/K_$(Prj_names[iPrj]).pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

figure()
iPrj = 6
K = readdlm(Prj_names[iPrj] * K_name)
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
scatter(x_inj, z_inj, c="r", marker=">", s=64)
scatter(x_prod, z_prod, c="r", marker="<", s=64)
savefig("figures_summary/K_$(Prj_names[iPrj]).pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

figure()
iPrj = 7
K = readdlm(Prj_names[iPrj] * K_name)
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
scatter(x_inj, z_inj, c="r", marker=">", s=64)
scatter(x_prod, z_prod, c="r", marker="<", s=64)
savefig("figures_summary/K_$(Prj_names[iPrj]).pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

figure()
z_src = [75].* dz .+ dz/2.0 # single source
x_src = (5-1)ones(Int64, size(z_src)) .* dx .+ dx/2.0
iPrj = 2
K = readdlm(Prj_names[iPrj] * K_name)
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
scatter(x_inj, z_inj, c="r", marker=">", s=64)
scatter(x_prod, z_prod, c="r", marker="<", s=64)
savefig("figures_summary/K_$(Prj_names[iPrj]).pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

rc("axes", titlesize=14)
rc("axes", labelsize=14)
rc("xtick", labelsize=14)
rc("ytick", labelsize=14)
rc("legend", fontsize=14)
figure()
iPrj = 7
brie_coef = readdlm(Prj_names[iPrj] * "/brie_coef.txt")[:,2]./30.0
plot(0:100,[2;brie_coef], "k");grid(ls="--")
# plot(1:100, 3ones(100))
xlabel("Iterations")
ylabel("Brie model coefficient")
savefig("figures_summary/brie_coef_curve.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);