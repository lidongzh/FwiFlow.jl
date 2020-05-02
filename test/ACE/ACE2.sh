
for noise in 0.0 0.01 0.05 0.10
do 
mkdir $noise 
cp ACE.jl $noise 
cd $noise 
srun julia ACE.jl 0 0 0.01 $noise && srun julia ACE.jl 1 0 0.01 $noise &
done 
