cmake_minimum_required(VERSION 3.10)

project(AbnormalAnalysis)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 指定编译器进行交叉编译
# set(CMAKE_C_COMPILER /home/ljh/ljh_nfs/my_toolchain/host/bin/aarch64-buildroot-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER /home/ljh/ljh_nfs/my_toolchain/host/bin/aarch64-buildroot-linux-gnu-g++)


if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../utils/ utils.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../freeze_detect/ normal_det.out)

include_directories(${CMAKE_SOURCE_DIR}/../include)



set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(LIBRKNNRT_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/rknpu2/include)
set(LIBRKNNRT ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/rknpu2/Linux/aarch64/librknnrt.so)


# set(OpenCV_DIR /home/ljh/ljh_nfs/ItcMultiDetect/3rdparty/GCC_10_4/opencv/opencv-linux-aarch64/share/OpenCV)
set(OpenCV_DIR /home/firefly/ljh/opencv4.5-dynamic/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)


file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${PROJECT_NAME} 
    main.cc
    rknpu2/resnet.cc
    resize_function.cc
    AbnormalAnalysis.cc
)


#.so
# add_library(${PROJECT_NAME} SHARED
#     rknpu2/resnet.cc
#     resize_function.cc
#     AbnormalAnalysis.cc  
# )


# 将 子目录 的库链接到主项目
target_link_libraries(${PROJECT_NAME}
    abnormal_detect
    fileutils
    imageutils
    imagedrawing
    ${LIBRKNNRT}
    ${OpenCV_LIBS}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME} log )
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBABNORMAL_INCLUDES}
)

# install(TARGETS ${PROJECT_NAME} DESTINATION .)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/dog_224x224.jpg DESTINATION model)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/synset.txt DESTINATION model)
# set(file_path ${CMAKE_CURRENT_SOURCE_DIR}/../model/resnet50-v2-7.rknn)
# if (EXISTS ${file_path})
#     install(FILES ${file_path} DESTINATION model)
# endif()
