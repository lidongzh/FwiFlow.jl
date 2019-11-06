import os

version = 'CO2_patchy_pgs'
gpuIds = '1_2'

if not os.path.exists(version):
    os.makedirs(version)

# cmd = 'julia main_two_phase_flow_inversion.jl --version=' + version + ' --generate_data=true --gpuIds=' + gpuIds + ' > ' + version + '/logGeneData.txt'

# print(cmd)
# os.system(cmd)

for indStage in range(2, 12):

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Stage ' + str(indStage))
    cmd = 'julia main_two_phase_flow_inversion.jl --version=' + version + ' --generate_data=false --indStage=' + str(indStage) + ' --gpuIds=' + gpuIds + ' > ' + version + '/log' + str(indStage) + '.txt'
    print(cmd)
    os.system(cmd)

