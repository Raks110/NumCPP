cmake_minimum_required(VERSION 3.15)
project(NumCPP)

set(CMAKE_CXX_STANDARD 14)

add_library(NumCPP STATIC numcpp.cpp numcpp.h parallel.h matrix.h)

include_directories($ENV{OPENCL_INCLUDE})
target_link_libraries(NumCPP $ENV{OPENCL_LIB})

add_custom_command(
        TARGET NumCPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/cmake-build-debug/src/libNumCPP.a
        ${CMAKE_SOURCE_DIR}/test/tmp/libNumCPP.a
)