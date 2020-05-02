#-------------------------------------------------------------------------------------
include_directories(${Src}/rapidjson)
add_subdirectory(${Src})

#-------------------------------------------------------------------------------------
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
target_link_libraries(FwiOp ${TF_LIB_FILE} CUFD)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set_target_properties(FwiOp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

