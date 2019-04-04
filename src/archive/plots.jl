using PyPlot
using DelimitedFiles
using Glob
using PyCall
@pyimport matplotlib2tikz as mpl

files = readdir(".")
loss = []
for i = 1:44
    for f in files
        if occursin("l", f) && occursin("_$i.txt", f)
            @show f
            push!(loss, readdlm(f)[1,1])
        end
    end
end
semilogy(loss)
xlabel("evaluation")
ylabel("loss")
mpl.save("loss.tex")

a = []
b1 = []
b2 = []
for i = 1:44
    for f in files
        if occursin("param", f) && occursin("_$i.txt", f)
            @show f
            val = readdlm(f)
            push!(a, 100*val[1])
            push!(b1, 1.0*val[2])
            push!(b2, -10.0*val[3])
        end
    end
end

N = length(a)
close("all")
figure(figsize=[9,3])
subplot(131)
plot(1:N, a, label="a")
plot(1:N, 100*ones(N), "--")
legend()
xlabel("evaluation")
ylabel("value")
subplot(132)
plot(1:N, b1, label="\$b_1\$")
plot(1:N, 1.0*ones(N), "--")
legend()
xlabel("evaluation")
ylabel("value")
subplot(133)
plot(1:N, b2, label="\$b_2\$")
plot(1:N, -10.0*ones(N), "--")
legend()
xlabel("evaluation")
ylabel("value")

subplots_adjust(wspace=0.5)