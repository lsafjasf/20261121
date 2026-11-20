cmake_minimum_required(VERSION 3.16)

project(true_peak_limiter VERSION 1.0.0 LANGUAGES CXX)

add_library(true_peak_limiter INTERFACE)
add_library(tpl::true_peak_limiter ALIAS true_peak_limiter)
target_include_directories(true_peak_limiter INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(true_peak_limiter INTERFACE cxx_std_17)

add_executable(peak_limiter_offline src/offline_main.cpp)
target_link_libraries(peak_limiter_offline PRIVATE tpl::true_peak_limiter)
target_include_directories(peak_limiter_offline PRIVATE src)

include(CTest)
if(BUILD_TESTING)
    add_executable(limiter_tests tests/core_tests.cpp)
    target_link_libraries(limiter_tests PRIVATE tpl::true_peak_limiter)
    target_include_directories(limiter_tests PRIVATE src)
    add_test(NAME limiter_tests COMMAND limiter_tests)
endif()

install(TARGETS peak_limiter_offline RUNTIME DESTINATION bin)
install(FILES include/TruePeakLimiter.hpp DESTINATION include)
