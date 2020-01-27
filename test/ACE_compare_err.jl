using MAT
using PyPlot
using PyCall 

mpl = pyimport("tikzplotlib")
# Error plot 
errw = []
erro = []
ss = [0.01,0.05,0.1,0.5]
d = matread("datamode0sparsity0.01/invData.mat")
errw = d["errw"][:]/10
erro = d["erro"][:]/10

close("all")
semilogy(errw, label="\$S_1\$")
semilogy(erro, label="\$S_2\$")
xlabel("Iteration")
ylabel("Error")
legend()
mpl.save("errwo.tex")
savefig("errwo.png")

# Estimation plot 
function plot_kr(krw, kro, w=missing, o=missing)
    close("all")
    plot(LinRange(0,1,100), krw, "b-", label="\$K_{rw}\$")
    plot(LinRange(0,1,100), kro, "r-", label="\$K_{ro}\$")
    if !ismissing(w)
        plot(LinRange(0,1,100), w, "g--", label="True \$K_{rw}\$")
        plot(LinRange(0,1,100), o, "c--", label="True \$K_{ro}\$")
    end
    xlabel("\$S_w\$")
    ylabel("\$K_r\$")
    legend()
end
d = matread("datamode0sparsity0.01/invData.mat")
dd = matread("datamode0sparsity0.01/Data.mat")
w = d["w491"][:]
o = d["o491"][:]
wref = dd["krw"][:]
oref = dd["kro"][:]
close("all");plot_kr(w, o, wref, oref); savefig("krwo.png")
mpl.save("krwo.tex")

rc("axes", titlesize=20)
rc("axes", labelsize=18)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("legend", fontsize=20)
# true model

figure()
m = 15
n = 30
h = 30
z_inj = (Int(round(0.6*m))-1)*h + h/2.0
x_inj = (Int(round(0.1*n))-1)*h + h/2.0
z_prod = (Int(round(0.6*m))-1)*h + h/2.0
x_prod = (Int(round(0.9*n))-1)*h + h/2.0
K = 20.0 .* ones(m,n)
K[8:10,:] .= 120.0
imshow(K, extent=[0,n*h,m*h,0]);
xlabel("Distance (m)")
ylabel("Depth (m)")
cb = colorbar()
clim([20, 120])
cb.set_label("Permeability (md)")
scatter(x_inj, z_inj, c="r", marker=">", s=64)
scatter(x_prod, z_prod, c="r", marker="<", s=64)
savefig("K.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);
