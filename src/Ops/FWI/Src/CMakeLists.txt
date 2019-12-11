cmake_minimum_required(VERSION 3.14)

# change to your own cuda compiler
set (CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-std=gnu++11 -O3")
project(CUFD)
include_directories(rapidjson)

find_package(CUDA QUIET)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-shared ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

cuda_add_library(CUFD SHARED Parameter.cpp libCUFD.cu el_stress.cu el_velocity.cu
ac_pressure.cu ac_velocity.cu Model.cu Cpml.cu utilities.cu 
Src_Rec.cu Boundary.cu ac_pressure_adj.cu ac_velocity_adj.cu 
image_vel.cu image_vel_time.cu ac_adj_push.cu el_stress_adj.cu 
el_velocity_adj.cu)
target_link_libraries(CUFD nvrtc cuda cudart cufft)
set_property(TARGET CUFD PROPERTY POSITION_INDEPENDENT_CODE ON)
