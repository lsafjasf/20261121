cmake_minimum_required(VERSION 3.16)
project(footbatch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(footbatch
    src/main.cpp
    src/json.cpp
    src/wav.cpp
    src/strategy.cpp
    src/library.cpp
    src/renderer.cpp
)
target_include_directories(footbatch PRIVATE include)

if(MSVC)
    target_compile_options(footbatch PRIVATE /W4)
else()
    target_compile_options(footbatch PRIVATE -Wall -Wextra -Wpedantic)
endif()
