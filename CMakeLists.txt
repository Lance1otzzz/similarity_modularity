cmake_minimum_required(VERSION 3.10)
project(test_project)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(main main.cpp graph.hpp defines.hpp)
add_executable(test test.cpp graph.hpp defines.hpp)

target_compile_options(main PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-Og -g>
)

target_compile_options(test PRIVATE -O3)