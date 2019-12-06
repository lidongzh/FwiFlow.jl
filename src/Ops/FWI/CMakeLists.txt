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

find_program (HOSTNAME_CMD NAMES hostname)
execute_process (COMMAND ${HOSTNAME_CMD}
                 OUTPUT_VARIABLE HOSTNAME
                 OUTPUT_STRIP_TRAILING_WHITESPACE)
## -- Set hostname ID (e.g., alcf, nwsc, nersc, ...)
message ("hostname is ${HOSTNAME}")
if (HOSTNAME MATCHES "Dolores")
  link_directories(Src/build_dolores)
else ()
  link_directories(Src/build)
endif()

add_library(FwiOp SHARED FwiOp.cpp)
set_property(TARGET FwiOp PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(FwiOp tensorflow_framework)
target_link_libraries(FwiOp CUFD)
