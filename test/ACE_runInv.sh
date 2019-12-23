for datamode in 0 1
do 
for sparsity in 0.5 0.1 0.05 0.01
do
julia ACE.jl 1 $datamode $sparsity &
done
done
