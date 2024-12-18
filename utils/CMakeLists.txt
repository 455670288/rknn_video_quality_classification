cmake_minimum_required(VERSION 3.15)

project(rknn_model_zoo_utils)

add_library(fileutils STATIC
    file_utils.c
)
target_include_directories(fileutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(imagedrawing STATIC
    image_drawing.c
)
target_include_directories(imagedrawing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# only RGA on rv1106 and rk3588 support handle
if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rk3588")
    add_definitions(-DLIBRGA_IM2D_HANDLE)
endif()


set(STB_INCLUDES ${CMAKE_SOURCE_DIR}/../3rdparty/stb_image)

set(LIBJPEG_INCLUDES ${CMAKE_SOURCE_DIR}/../3rdparty/jpeg_turbo/include)
set(LIBJPEG ${LIBJPEG_INCLUDES}/../Linux/aarch64/libturbojpeg.a)

set(LIBRGA_INCLUDES ${CMAKE_SOURCE_DIR}/../3rdparty/librga/include)
set(LIBRGA ${LIBRGA_INCLUDES}/../Linux/aarch64/librga.a)

add_library(imageutils STATIC
    image_utils.c
)
target_include_directories(imageutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)


target_link_libraries(imageutils
    ${LIBJPEG}
    ${LIBRGA}
)


target_include_directories(imageutils PUBLIC
    ${STB_INCLUDES}
    ${LIBJPEG_INCLUDES}
    ${LIBRGA_INCLUDES}
)