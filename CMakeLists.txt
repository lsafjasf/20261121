cmake_minimum_required(VERSION 3.16)

project(duplex_aec_debug LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(aec_core
    src/wav_io.cpp
    src/demo.cpp
    src/aec.cpp
    src/report.cpp
)

target_include_directories(aec_core PUBLIC src)
target_compile_options(aec_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(duplex_aec_debug src/main.cpp)
target_link_libraries(duplex_aec_debug PRIVATE aec_core)
target_compile_options(duplex_aec_debug PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME demo_smoke
        COMMAND duplex_aec_debug --demo --out-dir ${CMAKE_BINARY_DIR}/demo_smoke
    )
    set_tests_properties(demo_smoke PROPERTIES TIMEOUT 120
        FIXTURES_SETUP demo_audio)
    add_test(NAME external_smoke
        COMMAND duplex_aec_debug
            --far ${CMAKE_BINARY_DIR}/demo_smoke/far.wav
            --mic ${CMAKE_BINARY_DIR}/demo_smoke/mic.wav
            --near-clean ${CMAKE_BINARY_DIR}/demo_smoke/near_clean.wav
            --true-echo ${CMAKE_BINARY_DIR}/demo_smoke/true_echo.wav
            --dt-start 10 --dt-end 14
            --out-dir ${CMAKE_BINARY_DIR}/external_smoke
    )
    set_tests_properties(external_smoke PROPERTIES TIMEOUT 120
        FIXTURES_REQUIRED demo_audio)
endif()
