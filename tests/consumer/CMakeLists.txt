cmake_minimum_required(VERSION 3.24)
project(AppleSiliconFP64Consumer LANGUAGES ${APPLE_FP64_CONSUMER_LANGUAGE})

include(CTest)
if(APPLE_FP64_SOURCE_DIR)
    add_subdirectory("${APPLE_FP64_SOURCE_DIR}" apple-fp64)
    if(TARGET fp64_metal OR TARGET cpp_matrix_product)
        message(FATAL_ERROR "依存先のライブラリーがコマンドまたはテストを追加しました。")
    endif()
else()
    find_package(AppleSiliconFP64 CONFIG REQUIRED)
    cmake_path(IS_PREFIX CMAKE_PREFIX_PATH "${AppleSiliconFP64_METALLIB}" NORMALIZE resource_in_package)
    if(NOT resource_in_package)
        message(FATAL_ERROR "Metalライブラリーが移動後のパッケージを参照していません。")
    endif()
    get_target_property(include_directories AppleSiliconFP64::apple_fp64 INTERFACE_INCLUDE_DIRECTORIES)
    foreach(directory IN LISTS include_directories)
        cmake_path(IS_PREFIX CMAKE_PREFIX_PATH "${directory}" NORMALIZE include_in_package)
        if(NOT include_in_package)
            message(FATAL_ERROR "公開ヘッダーが移動後のパッケージを参照していません。")
        endif()
    endforeach()
endif()

add_executable(matrix_product matrix_product.c)
set_source_files_properties(matrix_product.c PROPERTIES LANGUAGE ${APPLE_FP64_CONSUMER_LANGUAGE})
target_link_libraries(matrix_product PRIVATE AppleSiliconFP64::apple_fp64)
add_custom_command(TARGET matrix_product POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${AppleSiliconFP64_METALLIB}"
        "$<TARGET_FILE_DIR:matrix_product>/fp64.metallib"
    VERBATIM)
add_test(NAME consumer_matrix_product COMMAND matrix_product "$<TARGET_FILE_DIR:matrix_product>/fp64.metallib")
