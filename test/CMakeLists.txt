set(BINARY ${CMAKE_PROJECT_NAME}_test)
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)
set(SOURCES ${TEST_SOURCES})
add_executable(${BINARY} main.cpp MatrixTest.cpp MatrixStatusTest.cpp MatrixOpsTest.cpp)
add_test(NAME ${BINARY} COMMAND ${BINARY})

set(PATH_TO_GOOGLETEST ./lib)

add_subdirectory(${PATH_TO_GOOGLETEST})
include_directories(${PATH_TO_GOOGLETEST}/googletest/include)
include_directories(${PATH_TO_GOOGLETEST}/googlemock/include)

target_link_libraries(${BINARY} gtest gtest_main)
target_link_libraries(${BINARY} gmock gmock_main)

include_directories(../src)
target_link_libraries(${BINARY} ${CMAKE_SOURCE_DIR}/test/tmp/libNumCPP.a)

include_directories($ENV{OPENCL_INCLUDE})
target_link_libraries(${BINARY} $ENV{OPENCL_LIB})