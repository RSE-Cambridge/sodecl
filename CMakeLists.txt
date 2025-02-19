#---------------------------------------------------------------------------//
# Copyright (c) 2015 Eleftherios Avramidis <el.avramidis@gmail.com>
#
# Distributed under The MIT License (MIT)
# See accompanying file LICENSE.txt
#---------------------------------------------------------------------------//

cmake_minimum_required (VERSION 3.13.4)
project(sodecl)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

set(default_build_type "Release")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	message(STATUS "USING OPTIONS FOR GNU COMPILER")
	set(MY_DEBUG_OPTIONS "-O0;-g;-Wall;-Wextra;-Wpedantic")
	set(MY_RELEASE_OPTIONS "-O3;-Wall;-Wextra;-Wpedantic")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	message(STATUS "USING OPTIONS FOR MSVC COMPILER")
	set(MY_DEBUG_OPTIONS "/Wall /wd4710")
	set(MY_RELEASE_OPTIONS "/Wall;/wd4710;/wd4711;/wd4514")
endif ()

add_compile_options("$<$<CONFIG:DEBUG>:${MY_DEBUG_OPTIONS}>")
add_compile_options("$<$<CONFIG:RELEASE>:${MY_RELEASE_OPTIONS}>")

FIND_PACKAGE(OpenCL REQUIRED)

add_subdirectory(benchmarks)
enable_testing()
add_subdirectory(tests)
add_subdirectory(interfaces)
add_subdirectory(examples)