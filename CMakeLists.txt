cmake_minimum_required(VERSION 3.24)
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0 CACHE STRING "実行するmacOSの最低バージョン" FORCE)
endif()
project(AppleSiliconFP64 LANGUAGES CXX OBJCXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(FATAL_ERROR "Apple SiliconとmacOSが必要です。")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "ビルド構成" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

find_program(XCRUN xcrun REQUIRED)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fp64.metallib
    COMMAND ${XCRUN} -sdk macosx metal -std=metal4.0 -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET} -I ${CMAKE_CURRENT_SOURCE_DIR}/src
        -c ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.metal -o ${CMAKE_CURRENT_BINARY_DIR}/fp64.air
    COMMAND ${XCRUN} -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/fp64.air
        -o ${CMAKE_CURRENT_BINARY_DIR}/fp64.metallib
    DEPENDS src/kernels.metal src/arithmetic.h
    VERBATIM)
add_custom_target(fp64_shaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fp64.metallib)

add_library(apple_fp64 STATIC src/matmul.mm)
target_include_directories(apple_fp64 PUBLIC include PRIVATE src)
target_compile_options(apple_fp64 PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(apple_fp64 PUBLIC "-framework Metal" "-framework Foundation")
add_dependencies(apple_fp64 fp64_shaders)

add_executable(fp64_metal src/main.cpp)
target_compile_definitions(fp64_metal PRIVATE ACCELERATE_NEW_LAPACK)
target_compile_options(fp64_metal PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(fp64_metal PRIVATE apple_fp64 "-framework Accelerate")

enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME exact_matrix_products COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify.py $<TARGET_FILE:fp64_metal>)

add_executable(repeated_matrix_products tests/repeated_matrix_products.cpp)
target_compile_options(repeated_matrix_products PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(repeated_matrix_products PRIVATE apple_fp64)
add_test(NAME repeated_matrix_products COMMAND repeated_matrix_products ${CMAKE_CURRENT_BINARY_DIR}/fp64.metallib)
