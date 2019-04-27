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

# include_directories(/home/huq11/anaconda3/lib/python3.6/site-packages/tensorflow/include)
# link_directories(/home/huq11/anaconda3/lib/python3.6/site-packages/tensorflow)
include_directories(/home/lidongzh/Library/anaconda3/envs/julia/lib/python3.6/site-packages/tensorflow/include)
link_directories(/home/lidongzh/Library/anaconda3/envs/julia/lib/python3.6/site-packages/tensorflow)
link_directories(Src/build)
add_library(FwiOp SHARED FwiOp.cpp)
set_property(TARGET FwiOp PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(FwiOp tensorflow_framework)
target_link_libraries(FwiOp CUFD)
