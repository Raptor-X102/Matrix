cmake_minimum_required(VERSION 3.20)
project(MatrixProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER_DIR ${CMAKE_SOURCE_DIR}/headers)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

include_directories(${HEADER_DIR})

option(BUILD_CLASSIC_TEST "Build classic Matrix test" ON)
option(ENABLE_TIME_MEASURE "Enable time measurement" OFF)
option(BUILD_ARITHMETIC_TEST "Build arithmetic operations test" ON)
option(BUILD_COMPLEX_TEST "Build complex number determinant test" ON)
option(BUILD_INVERSE_COMPARISON_TEST "Build inverse matrix comparison test" ON)
option(BUILD_UNIVERSAL_DETERMINANT_TEST "Build universal determinant test" ON)
option(BUILD_MIXED_MULTIPLICATION_TEST "Build mixed type multiplication test" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_SANITIZE "-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")

if(BUILD_CLASSIC_TEST)
    add_executable(matrix_classic_test_double ${SRC_DIR}/main_double.cpp)
    target_compile_definitions(matrix_classic_test_double PRIVATE MATRIX_IMPL_CLASSIC=1 USE_DOUBLE=1)

    add_executable(matrix_classic_test_int ${SRC_DIR}/main_int.cpp)
    target_compile_definitions(matrix_classic_test_int PRIVATE MATRIX_IMPL_CLASSIC=1 USE_INT=1)

    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_classic_test_double PRIVATE TIME_MEASURE=1)
        target_compile_definitions(matrix_classic_test_int PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_ARITHMETIC_TEST)
    add_executable(matrix_arithmetic_test ${SRC_DIR}/main.cpp)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_arithmetic_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_ARITHMETIC_TEST)
    add_executable(matrix_arithmetic_test_no_avx ${SRC_DIR}/main.cpp)
    set_property(TARGET matrix_arithmetic_test_no_avx PROPERTY CXX_STANDARD 17)
    target_compile_options(matrix_arithmetic_test_no_avx PRIVATE $<$<CONFIG:RELEASE>:-O3 -DNDEBUG>)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_arithmetic_test_no_avx PRIVATE TIME_MEASURE=1)
    endif()

    add_executable(matrix_arithmetic_test_with_avx ${SRC_DIR}/main.cpp)
    set_property(TARGET matrix_arithmetic_test_with_avx PROPERTY CXX_STANDARD 17)
    target_compile_options(matrix_arithmetic_test_with_avx PRIVATE $<$<CONFIG:RELEASE>:-O3 -DNDEBUG -mavx>)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_arithmetic_test_with_avx PRIVATE TIME_MEASURE=1)
    endif()

    add_executable(matrix_multiply_benchmark ${SRC_DIR}/main_benchmark.cpp)
    set_property(TARGET matrix_multiply_benchmark PROPERTY CXX_STANDARD 17)
    target_compile_options(matrix_multiply_benchmark PRIVATE $<$<CONFIG:RELEASE>:-O3 -DNDEBUG -mavx>)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_multiply_benchmark PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_COMPLEX_TEST)
    add_executable(matrix_complex_det_test ${SRC_DIR}/main_complex.cpp)
    set_property(TARGET matrix_complex_det_test PROPERTY CXX_STANDARD 17)
    target_compile_options(matrix_complex_det_test PRIVATE $<$<CONFIG:RELEASE>:-O3 -DNDEBUG -mavx>)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_complex_det_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_INVERSE_COMPARISON_TEST)
    add_executable(matrix_inverse_comparison_test ${SRC_DIR}/main_inverse_comparison.cpp)
    set_property(TARGET matrix_inverse_comparison_test PROPERTY CXX_STANDARD 17)
    target_compile_options(matrix_inverse_comparison_test PRIVATE $<$<CONFIG:RELEASE>:-O3 -DNDEBUG -mavx>)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_inverse_comparison_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_UNIVERSAL_DETERMINANT_TEST)
    add_executable(matrix_universal_det_test ${SRC_DIR}/main_universal_det.cpp)
    set_property(TARGET matrix_universal_det_test PROPERTY CXX_STANDARD 17)
    target_compile_options(matrix_universal_det_test PRIVATE $<$<CONFIG:RELEASE>:-O3 -DNDEBUG -mavx>)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_universal_det_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_MIXED_MULTIPLICATION_TEST)
    add_executable(matrix_mixed_multiplication_test ${SRC_DIR}/main_test_multiplication.cpp)
    set_property(TARGET matrix_mixed_multiplication_test PROPERTY CXX_STANDARD 17)
    target_compile_options(matrix_mixed_multiplication_test PRIVATE $<$<CONFIG:RELEASE>:-O3 -DNDEBUG -mavx>)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_mixed_multiplication_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Sanitize")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SANITIZE}")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Time measurement: ${ENABLE_TIME_MEASURE}")
message(STATUS "Build classic test: ${BUILD_CLASSIC_TEST}")
message(STATUS "Build arithmetic test: ${BUILD_ARITHMETIC_TEST}")
message(STATUS "Build complex test: ${BUILD_COMPLEX_TEST}")
message(STATUS "Build inverse comparison test: ${BUILD_INVERSE_COMPARISON_TEST}")
message(STATUS "Build universal determinant test: ${BUILD_UNIVERSAL_DETERMINANT_TEST}")
message(STATUS "Build mixed multiplication test: ${BUILD_MIXED_MULTIPLICATION_TEST}")

find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    set(CLANG_FORMAT_FILE "${CMAKE_SOURCE_DIR}/.clang-format")
    if(EXISTS "${CLANG_FORMAT_FILE}")
        file(GLOB_RECURSE ALL_SOURCE_FILES
            "${SRC_DIR}/*.cpp"
            "${SRC_DIR}/*.c"
            "${HEADER_DIR}/*.h"
            "${HEADER_DIR}/*.hpp"
        )
        if(ALL_SOURCE_FILES)
            add_custom_target(format
                COMMAND ${CLANG_FORMAT} -i --style=file:${CLANG_FORMAT_FILE} ${ALL_SOURCE_FILES}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                COMMENT "Formatting source files with clang-format"
            )
        endif()
    endif()
endif()
