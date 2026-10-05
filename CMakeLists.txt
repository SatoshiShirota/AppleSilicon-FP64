cmake_minimum_required(VERSION 3.24)
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
endif()
project(AppleSiliconFP64 LANGUAGES C OBJC)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(FATAL_ERROR "Apple SiliconとmacOSが必要です。")
endif()
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "ビルド構成" FORCE)
endif()
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_OBJC_STANDARD 11)
set(CMAKE_OBJC_STANDARD_REQUIRED ON)
set(CMAKE_OBJC_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif()
option(APPLE_FP64_BUILD_TOOLS "実験用コマンドをビルドする" ${PROJECT_IS_TOP_LEVEL})

set(apple_fp64_metallib_name fp64.metallib)
set(AppleSiliconFP64_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/${apple_fp64_metallib_name}")
if(NOT PROJECT_IS_TOP_LEVEL)
    set(AppleSiliconFP64_METALLIB "${AppleSiliconFP64_METALLIB}" PARENT_SCOPE)
endif()
find_program(XCRUN xcrun REQUIRED)
add_custom_command(OUTPUT "${AppleSiliconFP64_METALLIB}"
    COMMAND ${XCRUN} -sdk macosx metal -std=metal4.0 -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET} -I ${CMAKE_CURRENT_SOURCE_DIR}/src
        -c ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.metal -o ${CMAKE_CURRENT_BINARY_DIR}/fp64.air
    COMMAND ${XCRUN} -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/fp64.air
        -o "${AppleSiliconFP64_METALLIB}"
    DEPENDS src/kernels.metal src/arithmetic.h src/fp64_fma.h
    VERBATIM)
add_custom_target(fp64_shaders DEPENDS "${AppleSiliconFP64_METALLIB}")

add_library(apple_fp64 STATIC src/matmul.m)
add_library(AppleSiliconFP64::apple_fp64 ALIAS apple_fp64)
target_include_directories(apple_fp64
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE src)
target_compile_options(apple_fp64 PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(apple_fp64 PUBLIC "-framework Metal" "-framework Foundation")
add_dependencies(apple_fp64 fp64_shaders)

set(apple_fp64_install_resources "${CMAKE_INSTALL_DATADIR}/AppleSiliconFP64")
set(apple_fp64_install_metallib "${apple_fp64_install_resources}/${apple_fp64_metallib_name}")
set(apple_fp64_install_cmake "${CMAKE_INSTALL_LIBDIR}/cmake/AppleSiliconFP64")
configure_package_config_file(cmake/AppleSiliconFP64Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/AppleSiliconFP64Config.cmake"
    INSTALL_DESTINATION "${apple_fp64_install_cmake}"
    PATH_VARS apple_fp64_install_metallib)
install(TARGETS apple_fp64 EXPORT AppleSiliconFP64Targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES include/apple_fp64/matmul.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/apple_fp64")
install(FILES "${AppleSiliconFP64_METALLIB}" DESTINATION "${apple_fp64_install_resources}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/AppleSiliconFP64")
install(EXPORT AppleSiliconFP64Targets NAMESPACE AppleSiliconFP64::
    DESTINATION "${apple_fp64_install_cmake}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/AppleSiliconFP64Config.cmake"
    DESTINATION "${apple_fp64_install_cmake}")

if(APPLE_FP64_BUILD_TOOLS OR (PROJECT_IS_TOP_LEVEL AND BUILD_TESTING))
    add_executable(fp64_metal src/main.c)
    target_compile_definitions(fp64_metal PRIVATE ACCELERATE_NEW_LAPACK)
    target_compile_options(fp64_metal PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(fp64_metal PRIVATE apple_fp64 "-framework Accelerate")
endif()

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME exact_matrix_products COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify.py $<TARGET_FILE:fp64_metal>)
    add_test(NAME benchmark_measurements COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_measurements.py $<TARGET_FILE:fp64_metal>)

    add_executable(fused_multiply_add tests/fused_multiply_add.c)
    target_include_directories(fused_multiply_add PRIVATE src)
    target_compile_options(fused_multiply_add PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME fused_multiply_add COMMAND fused_multiply_add)

    add_executable(repeated_matrix_products tests/repeated_matrix_products.c)
    target_compile_options(repeated_matrix_products PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(repeated_matrix_products PRIVATE apple_fp64)
    add_test(NAME repeated_matrix_products COMMAND repeated_matrix_products "${AppleSiliconFP64_METALLIB}")

    enable_language(CXX)
    add_executable(cpp_matrix_product tests/cpp_matrix_product.cpp)
    set_target_properties(cpp_matrix_product PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS OFF)
    target_compile_options(cpp_matrix_product PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(cpp_matrix_product PRIVATE apple_fp64)
    add_test(NAME cpp_matrix_product COMMAND cpp_matrix_product "${AppleSiliconFP64_METALLIB}")

    add_test(NAME library_consumers COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/library_consumers.py
        "${CMAKE_COMMAND}" "${CMAKE_CTEST_COMMAND}"
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
        "$<CONFIG>" "${CMAKE_GENERATOR}")
    set_tests_properties(library_consumers PROPERTIES TIMEOUT 180)
endif()
