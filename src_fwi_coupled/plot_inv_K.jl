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
Prj_names = ["CO2", "CO2_1src", "CO2_2surveys"]
K_name = "/K$iter.txt"

# true model
figure()
K = 20.0 .* ones(m,n)
K[8:10,:] .= 120.0
imshow(K, extent=[0,n*h,m*h,0]);
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
shot_inds = collect(1:length(z_src))
# scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
# scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
# scatter(x_inj, z_inj, c="r", marker=">")
# scatter(x_prod, z_prod, c="r", marker="<")
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
# scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
# scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
# scatter(x_inj, z_inj, c="r", marker=">")
# scatter(x_prod, z_prod, c="r", marker="<")
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
scatter(x_inj, z_inj, c="r", marker=">")
scatter(x_prod, z_prod, c="r", marker="<")
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
scatter(x_inj, z_inj, c="r", marker=">")
scatter(x_prod, z_prod, c="r", marker="<")
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
scatter(x_inj, z_inj, c="r", marker=">")
scatter(x_prod, z_prod, c="r", marker="<")
savefig("figures_summary/K_$(Prj_names[iPrj]).pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);