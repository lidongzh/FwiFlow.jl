using DelimitedFiles
using PyPlot

close("all")

if !isdir("figures_summary")
  mkdir("figures_summary")
end

Prj_names = ["CO2", "CO2_1src", "CO2_2surveys"]

L1 = readdlm("$(Prj_names[1])/loss.txt")
l1=semilogy(L1[:,1], L1[:,2]/L1[1,2], label="Baseline")
legend()

L2 = readdlm("$(Prj_names[2])/loss.txt")
l2=semilogy(L2[:,1], L2[:,2]/L2[1,2], label="One source")
legend()

L3 = readdlm("$(Prj_names[3])/loss.txt")
l3=semilogy(L3[:,1], L3[:,2]/L3[1,2], label="Two surveys")
legend()

xlabel("Iteration Number")
ylabel("Normalized misfit")

savefig("figures_summary/loss.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);