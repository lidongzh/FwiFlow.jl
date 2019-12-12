cmake_minimum_required(VERSION 3.5)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 ${CMAKE_CXX_FLAGS}")


execute_process(COMMAND julia -e "using Conda; print(joinpath(Conda.LIBDIR, \"Libraries\"))" OUTPUT_VARIABLE EIGEN_INC)
execute_process(COMMAND julia -e "using Conda; print(joinpath(Conda.PYTHONDIR, \"python\"))" OUTPUT_VARIABLE PYTHON)
execute_process(COMMAND julia -e "using Conda; print(joinpath(Conda.ROOTENV, \"include\"))" OUTPUT_VARIABLE CONDA_INC)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; \
                                       TF_INC = tf.sysconfig.get_compile_flags()[0][2:];\
                                       TF_LIB = tf.sysconfig.get_link_flags()[0][2:];\
                                       TF_ABI = tf.sysconfig.get_compile_flags()[1][-1];\
                                       TF_LIB_FILE = tf.sysconfig.get_link_flags()[1][3:];\
                                       OUT = ';'.join([TF_INC,TF_LIB,TF_ABI,TF_LIB_FILE]);\
                                       sys.stdout.write(OUT)" OUTPUT_VARIABLE PY_OUT)
list(GET PY_OUT 0 TF_INC)
list(GET PY_OUT 1 TF_LIB)
list(GET PY_OUT 2 TF_ABI)
list(GET PY_OUT 3 TF_LIB_FILE)


message("Python path=${PYTHON}")
message("EIGEN_INC=${EIGEN_INC}")
message("TF_INC=${TF_INC}")
message("TF_LIB=${TF_LIB}")
message("TF_ABI=${TF_ABI}")
message("TF_LIB_FILE=${TF_LIB_FILE}")
message("CONDA_INC=${CONDA_INC}")
# https://github.com/tensorflow/tensorflow/issues/1569
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()
message(${CMAKE_CXX_FLAGS})
include_directories(${TF_INC})
link_directories(${TF_LIB})
# --- Find AMGCL ------------------------------------------------------------
include_directories("../../amgcl")
include_directories(../../eigen)
include_directories(/usr/local/include ${CONDA_INC})

add_library(PoissonOp SHARED PoissonOp.cpp)
set_property(TARGET PoissonOp PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(PoissonOp tensorflow_framework)
