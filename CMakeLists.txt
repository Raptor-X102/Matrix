cmake_minimum_required(VERSION 3.20)
project(MatrixProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER_DIR ${CMAKE_SOURCE_DIR}/headers)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

include_directories(${HEADER_DIR})

add_executable(matrix_app ${SRC_DIR}/main.cpp)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(ENABLE_TIME_MEASURE "Enable time measurement" OFF)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_SANITIZE "-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
    target_compile_definitions(matrix_app PRIVATE DEBUG_MODE=1)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    target_compile_definitions(matrix_app PRIVATE RELEASE_MODE=1)
elseif(CMAKE_BUILD_TYPE STREQUAL "Sanitize")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SANITIZE}")
    target_compile_definitions(matrix_app PRIVATE SANITIZE_MODE=1)
endif()

if(ENABLE_TIME_MEASURE)
    target_compile_definitions(matrix_app PRIVATE TIME_MEASURE=1)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Time measurement: ${ENABLE_TIME_MEASURE}")
