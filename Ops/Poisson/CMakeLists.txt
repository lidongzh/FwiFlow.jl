cmake_minimum_required(VERSION 3.5)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# https://github.com/tensorflow/tensorflow/issues/1569
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1 ${CMAKE_CXX_FLAGS}")
endif()

# import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))
# import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))

include_directories(/home/lidongzh/Library/anaconda3/envs/julia/lib/python3.6/site-packages/tensorflow/include /home/lidongzh/Library/PyTensorFlow.jl/src/../examples/custom_op/template/../../../deps/ADEL/ADEL)
include_directories(/home/lidongzh/TwoPhaseFlowFWI/eigen)
link_directories(/home/lidongzh/Library/anaconda3/envs/julia/lib/python3.6/site-packages/tensorflow)

add_library(PoissonOp SHARED PoissonOp.cpp)
set_property(TARGET PoissonOp PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(PoissonOp tensorflow_framework)
