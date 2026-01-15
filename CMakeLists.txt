cmake_minimum_required(VERSION 3.20)
project(MatrixProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER_DIR ${CMAKE_SOURCE_DIR}/headers)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

include_directories(${HEADER_DIR})

# === Выбор теста: classic или 1d или оба ===
option(BUILD_CLASSIC_TEST "Build classic Matrix test" ON)
option(BUILD_1D_TEST "Build 1D Matrix test" ON)
option(ENABLE_TIME_MEASURE "Enable time measurement" OFF)

# === Флаги ===
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_SANITIZE "-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer")

# === Сборка тестов ===

if(BUILD_CLASSIC_TEST)
    add_executable(matrix_classic_test ${SRC_DIR}/main.cpp)
    target_compile_definitions(matrix_classic_test PRIVATE MATRIX_IMPL_CLASSIC=1)

    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_classic_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

if(BUILD_1D_TEST)
    add_executable(matrix_1d_test ${SRC_DIR}/main.cpp)
    target_compile_definitions(matrix_1d_test PRIVATE MATRIX_IMPL_1D=1)

    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_1d_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

# === Общий тест, сравнивающий обе реализации ===
if(BUILD_CLASSIC_TEST AND BUILD_1D_TEST)
    add_executable(matrix_compare_test ${SRC_DIR}/main.cpp)  # новый файл

    if(ENABLE_TIME_MEASURE)
        target_compile_definitions(matrix_compare_test PRIVATE TIME_MEASURE=1)
    endif()
endif()

# === Флаги сборки ===
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Sanitize")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SANITIZE}")
endif()

# === Вывод настроек ===
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Time measurement: ${ENABLE_TIME_MEASURE}")
message(STATUS "Build classic test: ${BUILD_CLASSIC_TEST}")
message(STATUS "Build 1D test: ${BUILD_1D_TEST}")
