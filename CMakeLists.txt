cmake_minimum_required(VERSION 3.10)
project(MatrixProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER_DIR ${CMAKE_SOURCE_DIR}/headers)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

include_directories(${HEADER_DIR})
include_directories(${HEADER_DIR}/Matrix)
include_directories(${HEADER_DIR}/Vector)

option(ENABLE_AVX "Enable AVX optimizations" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(ENABLE_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -march=native")
    add_definitions(-D__AVX__)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

add_executable(determinant_calculator ${SRC_DIR}/main_determinant.cpp)

file(GLOB TEST_FILES ${SRC_DIR}/tests/*.cpp)
set(TEST_TARGETS "")

foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    set_target_properties(${test_name} PROPERTIES CXX_STANDARD 17)
    list(APPEND TEST_TARGETS ${test_name})
endforeach()

add_custom_target(tests DEPENDS ${TEST_TARGETS})

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
