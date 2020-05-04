cmake_minimum_required(VERSION 3.5)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 ${CMAKE_CXX_FLAGS}")

execute_process(COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[0][2:])" OUTPUT_VARIABLE TF_INC)
execute_process(COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_link_flags()[0][2:])" OUTPUT_VARIABLE TF_LIB)
execute_process(COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[1][-1])" OUTPUT_VARIABLE TF_ABI)

# https://github.com/tensorflow/tensorflow/issues/1569
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()
message(${CMAKE_CXX_FLAGS})
include_directories(${TF_INC})
link_directories(${TF_LIB})

# --- Find AMGCL ------------------------------------------------------------
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

add_library(Eikonal SHARED Eikonal.cpp)
set_property(TARGET Eikonal PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Eikonal tensorflow_framework)
