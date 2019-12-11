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
include_directories(CONDA_INC)

message("Python path=${PYTHON}")
message("EIGEN_INC=${EIGEN_INC}")
message("TF_INC=${TF_INC}")
message("TF_LIB=${TF_LIB}")
message("TF_ABI=${TF_ABI}")
message("TF_LIB_FILE=${TF_LIB_FILE}")
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
include_directories(/usr/local/include)
# target_compile_options(eigen_target INTERFACE
#     $<$<CXX_COMPILER_ID:GNU>:-Wno-int-in-bool-context>
#     $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>
#     $<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-long-long>
#     )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(UpwpsOp SHARED UpwpsOp.cpp)
set_property(TARGET UpwpsOp PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(UpwpsOp tensorflow_framework)
