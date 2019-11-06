clear;
close all;

%%
nz = 150;
nx = 300;
model = 20 * ones(nz, nx);
x = 1:nx;
y1 = 60 + 10*sin(x/100*2*pi);
y2 = 100 + 10*sin(x/100*2*pi);
for j = 1:nx
    for i = 1:nz
        if (i > y1(j) && i < y2(j))
            model(i, j) = 120;
        end
    end
end

%%
imagesc(model);