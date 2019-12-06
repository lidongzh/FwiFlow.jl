using PyPlot
using DelimitedFiles

if !isdir("figures_summary_channel")
  mkdir("figures_summary_channel")
end

m = 90
n = 180
h = 5.0 # meter
dz = 3.0 # meters
dx = 3.0
nz = Int64(round((m * h) / dz)) + 1
nx = Int64(round((n * h) / dx)) + 1
z_src = (collect(5:10:nz-5) .- 1 ) .* dz .+ dz/2.0
x_src = (5-1)ones(Int64, size(z_src)) .* dx .+ dx/2.0
z_rec = (collect(5:1:nz-5) .- 1) .* dz .+ dz/2.0
x_rec = (nx-5-1) .* ones(Int64, size(z_rec)) .*dx .+ dx/2.0

z_inj = (54-1)*h + h/2.0
x_inj = (18-1)*h + h/2.0
z_prod = (54-1)*h + h/2.0
x_prod = (168-1)*h + h/2.0

rc("axes", titlesize=20)
rc("axes", labelsize=18)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("legend", fontsize=20)
# true model
figure()
    K = 20.0 .* ones(m,n) # millidarcy
    ix = 1:n
    y1 = 45. .+ 10. .* sin.(ix./120.0 .* 2.0 .* pi)
    y2 = 55. .+ 10. .* sin.(ix./120.0 .* 2.0 .* pi)
    for j = 1:n
        for i = 1:m
            if (i > y1[j] && i < y2[j])
                K[i, j] = 120;
            end
        end
    end
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
savefig("figures_summary_channel/K_true.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);


# # init model
# figure()
# K = 20.0 .* ones(m,n)
# imshow(K, extent=[0,n*h,m*h,0]);
# xlabel("Distance (m)")
# ylabel("Depth (m)")
# cb = colorbar()
# clim([20, 120])
# cb.set_label("Permeability (md)")
# shot_inds = collect(1:length(z_src))
# scatter(x_src[shot_inds], z_src[shot_inds], c="w", marker="*")
# scatter(x_rec, z_rec, s=16.0, c="r", marker="v")
# scatter(x_inj, z_inj, c="r", marker=">")
# scatter(x_prod, z_prod, c="r", marker="<")
# savefig("figures_summary/K_init.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);


m = 45
n = 90
h = 10.0 # meter
dz = 3.0 # meters
dx = 3.0
nz = Int64(round((m * h) / dz)) + 1
nx = Int64(round((n * h) / dx)) + 1
z_src = (collect(5:10:nz-5) .- 1 ) .* dz .+ dz/2.0
x_src = (5-1)ones(Int64, size(z_src)) .* dx .+ dx/2.0
z_rec = (collect(5:1:nz-5) .- 1) .* dz .+ dz/2.0
x_rec = (nx-5-1) .* ones(Int64, size(z_rec)) .*dx .+ dx/2.0

z_inj = (27-1)*h + h/2.0
x_inj = (9-1)*h + h/2.0
z_prod = (27-1)*h + h/2.0
x_prod = (84-1)*h + h/2.0
iter = 100
Prj_names = ["CO2_channel_4590_pgs", "CO2_channel_45_90"]
K_name = "/Stage6/K$iter.txt"
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
savefig("figures_summary_channel/K_$(Prj_names[iPrj]).pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

K_name = "/K$iter.txt"
figure()
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
savefig("figures_summary_channel/K_$(Prj_names[iPrj]).pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);