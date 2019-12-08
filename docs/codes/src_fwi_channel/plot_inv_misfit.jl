using DelimitedFiles
using PyPlot

close("all")

if !isdir("figures_summary")
  mkdir("figures_summary")
end

Prj_names = ["CO2", "CO2_1src", "CO2_2surveys", "Brie_3_nocoefupdate", "Brie_tune_coef_true3_start2"]

rc("axes", titlesize=14)
rc("axes", labelsize=14)
rc("xtick", labelsize=14)
rc("ytick", labelsize=14)
rc("legend", fontsize=14)

figure()
L1 = readdlm("$(Prj_names[1])/loss.txt")
l1=semilogy(L1[:,1], L1[:,2]/L1[1,2], label="Baseline")
legend()

L2 = readdlm("$(Prj_names[2])/loss.txt")
l2=semilogy(L2[:,1], L2[:,2]/L2[1,2], label="One source")
legend()

L3 = readdlm("$(Prj_names[3])/loss.txt")
l3=semilogy(L3[:,1], L3[:,2]/L3[1,2], label="Two surveys")
legend()

grid(ls="--")

xlabel("Iteration Number")
ylabel("Normalized misfit")

savefig("figures_summary/loss.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);


figure()
L4 = readdlm("$(Prj_names[4])/loss.txt")
l4=semilogy(L4[:,1], L4[:,2]/L4[1,2], label="Exact coefficient")
legend()

L5 = readdlm("$(Prj_names[5])/loss.txt")
l5=semilogy(L5[:,1], L5[:,2]/L5[1,2], label="Inexact coefficient")
legend()

grid(ls="--")

xlabel("Iteration Number")
ylabel("Normalized misfit")

savefig("figures_summary/loss_brie.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);