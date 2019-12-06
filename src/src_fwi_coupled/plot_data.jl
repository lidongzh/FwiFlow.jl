using PyPlot
using DelimitedFiles

if !isdir("figures_summary")
  mkdir("figures_summary")
end

m = 15
n = 30
h = 30.0 # meter
dt = 0.00025
nt = 3001

rc("axes", titlesize=16)
rc("axes", labelsize=12)
rc("xtick", labelsize=12)
rc("ytick", labelsize=12)
rc("legend", fontsize=16)

shot1=read("CO2/Data1/Shot8.bin");
shot1 = reshape(reinterpret(Float32,shot1),(nt,142))
fig,ax = subplots()
imshow(shot1, extent=[0,m*h,(nt-1)*dt,0], cmap="gray", aspect=1.5*(m*h)/((nt-1)*dt));
# imshow(shot1', extent=[0,(nt-1)*dt,m*h,0], cmap="gray", aspect=0.8*(nt-1)*dt/(m*h));
xlabel("Depth (m)")
ylabel("Time (s)")
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top") 
savefig("figures_summary/CO2_Data1_Shot8.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

shot2=read("CO2/Data11/Shot8.bin");
shot2 = reshape(reinterpret(Float32,shot2),(nt,142))
fig,ax = subplots()
imshow(shot2, extent=[0,m*h,(nt-1)*dt,0], cmap="gray", aspect=1.5*(m*h)/((nt-1)*dt));
xlabel("Depth (m)")
ylabel("Time (s)")
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top") 
savefig("figures_summary/CO2_Data11_Shot8.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);

fig,ax = subplots()
imshow(shot2-shot1, extent=[0,m*h,(nt-1)*dt,0], cmap="gray", aspect=1.5*(m*h)/((nt-1)*dt));
xlabel("Depth (m)")
ylabel("Time (s)")
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top") 
savefig("figures_summary/CO2_Data_diff.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);