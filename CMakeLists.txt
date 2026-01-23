cmake_minimum_required(VERSION 3.20)
project(MatrixProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER_DIR ${CMAKE_SOURCE_DIR}/headers)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# Включаем все поддиректории с заголовками
include_directories(${HEADER_DIR})
include_directories(${HEADER_DIR}/Matrix)
include_directories(${HEADER_DIR}/Vector)

# Настройки компиляции
option(ENABLE_AVX "Enable AVX optimizations" ON)
option(ENABLE_TIME_MEASURE "Enable time measurement" OFF)

# Опции сборки тестов
option(BUILD_CLASSIC_TEST "Build classic Matrix test" ON)
option(BUILD_ARITHMETIC_TEST "Build arithmetic operations test" ON)
option(BUILD_COMPLEX_TEST "Build complex number determinant test" ON)
option(BUILD_INVERSE_COMPARISON_TEST "Build inverse matrix comparison test" ON)
option(BUILD_UNIVERSAL_DETERMINANT_TEST "Build universal determinant test" ON)
option(BUILD_MIXED_MULTIPLICATION_TEST "Build mixed type multiplication test" ON)
option(BUILD_TRANSPOSE_TIMING_TEST "Build transpose timing test" ON)
option(BUILD_VECTOR_TEST "Build Vector class test" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Флаги компиляции
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_SANITIZE "-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer")

# Добавляем AVX флаги если включено
if(ENABLE_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -march=native")
    add_definitions(-D__AVX__)
endif()

if(BUILD_CLASSIC_TEST)
    add_executable(matrix_classic_test_double ${SRC_DIR}/main_double.cpp)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_classic_test_double PRIVATE TIME_MEASURE=1)
    endif()

    add_executable(matrix_classic_test_int ${SRC_DIR}/main_int.cpp)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_classic_test_int PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_ARITHMETIC_TEST)
    add_executable(matrix_arithmetic_test ${SRC_DIR}/main.cpp)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_arithmetic_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_COMPLEX_TEST)
    add_executable(matrix_complex_det_test ${SRC_DIR}/main_complex.cpp)
    set_property(TARGET matrix_complex_det_test PROPERTY CXX_STANDARD 17)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_complex_det_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_INVERSE_COMPARISON_TEST)
    add_executable(matrix_inverse_comparison_test ${SRC_DIR}/main_inverse_comparison.cpp)
    set_property(TARGET matrix_inverse_comparison_test PROPERTY CXX_STANDARD 17)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_inverse_comparison_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_UNIVERSAL_DETERMINANT_TEST)
    add_executable(matrix_universal_det_test ${SRC_DIR}/main_universal_det.cpp)
    set_property(TARGET matrix_universal_det_test PROPERTY CXX_STANDARD 17)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_universal_det_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_MIXED_MULTIPLICATION_TEST)
    add_executable(matrix_mixed_multiplication_test ${SRC_DIR}/main_test_multiplication.cpp)
    set_property(TARGET matrix_mixed_multiplication_test PROPERTY CXX_STANDARD 17)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_mixed_multiplication_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_TRANSPOSE_TIMING_TEST)
    add_executable(matrix_transpose_timing_test ${SRC_DIR}/main_transpose_timing.cpp)
    set_property(TARGET matrix_transpose_timing_test PROPERTY CXX_STANDARD 17)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_transpose_timing_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_VECTOR_TEST)
    add_executable(vector_test ${SRC_DIR}/Vector_test.cpp)
    set_property(TARGET vector_test PROPERTY CXX_STANDARD 17)
    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(vector_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

# Применяем флаги в зависимости от типа сборки
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Sanitize")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SANITIZE}")
endif()

# Вывод информации о конфигурации
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "AVX optimizations: ${ENABLE_AVX}")
message(STATUS "Time measurement: ${ENABLE_TIME_MEASURE}")
message(STATUS "Build classic test: ${BUILD_CLASSIC_TEST}")
message(STATUS "Build arithmetic test: ${BUILD_ARITHMETIC_TEST}")
message(STATUS "Build complex test: ${BUILD_COMPLEX_TEST}")
message(STATUS "Build inverse comparison test: ${BUILD_INVERSE_COMPARISON_TEST}")
message(STATUS "Build universal determinant test: ${BUILD_UNIVERSAL_DETERMINANT_TEST}")
message(STATUS "Build mixed multiplication test: ${BUILD_MIXED_MULTIPLICATION_TEST}")
message(STATUS "Build transpose timing test: ${BUILD_TRANSPOSE_TIMING_TEST}")
message(STATUS "Build vector test: ${BUILD_VECTOR_TEST}")

# Настройка clang-format если доступен
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    set(CLANG_FORMAT_FILE "${CMAKE_SOURCE_DIR}/.clang-format")
    if(EXISTS "${CLANG_FORMAT_FILE}")
        file(GLOB_RECURSE ALL_SOURCE_FILES
            "${SRC_DIR}/*.cpp"
            "${SRC_DIR}/*.c"
            "${HEADER_DIR}/*.h"
            "${HEADER_DIR}/*.hpp"
            "${HEADER_DIR}/Matrix/*.hpp"
            "${HEADER_DIR}/Matrix/*.ipp"
            "${HEADER_DIR}/Vector/*.hpp"
            "${HEADER_DIR}/Vector/*.ipp"
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
