cmake_minimum_required(VERSION 3.5)
project(TF_CUSTOM_OP)
set (CMAKE_CXX_STANDARD 11)

execute_process(COMMAND julia -e "import ADCME; print(ADCME.__STR__)" OUTPUT_VARIABLE JL_OUT)
list(GET JL_OUT 0 BINDIR)
list(GET JL_OUT 1 LIBDIR)
list(GET JL_OUT 2 TF_INC)
list(GET JL_OUT 3 TF_ABI)
list(GET JL_OUT 4 EIGEN_INC)
list(GET JL_OUT 5 CC)
list(GET JL_OUT 6 CXX)
list(GET JL_OUT 7 CMAKE)
list(GET JL_OUT 8 MAKE)
list(GET JL_OUT 9 GIT)
list(GET JL_OUT 10 PYTHON)
list(GET JL_OUT 11 TF_LIB_FILE)

message("Python path=${PYTHON}")
message("EIGEN_INC=${EIGEN_INC}")
message("TF_INC=${TF_INC}")
message("TF_ABI=${TF_ABI}")
message("TF_LIB_FILE=${TF_LIB_FILE}")


if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include_directories(${TF_INC} ${HEADERS} ${EIGEN_INC})
link_directories(${TF_LIB})
include_directories(${LIBDIR}/../include) # for boost


# --- Find AMGCL ------------------------------------------------------------
include_directories("../../amgcl")
include_directories(../../eigen)
include_directories(/usr/local/include ${CONDA_INC})
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
target_link_libraries(UpwpsOp ${TF_LIB_FILE})
