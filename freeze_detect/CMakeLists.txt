cmake_minimum_required(VERSION 3.4.1)

project(abnormal_detect)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -funroll-loops -march=native") #编译器生成针对主机架构优化的代码
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections") #链接器只包含实际使用的代码和数据,减小二进制文件大小


# 指定编译器进行交叉编译
# set(CMAKE_C_COMPILER /home/ljh/ljh_nfs/my_toolchain/host/bin/aarch64-buildroot-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER /home/ljh/ljh_nfs/my_toolchain/host/bin/aarch64-buildroot-linux-gnu-g++)


# set(CMAKE_BUILD_TYPE "Debug") #启用调试信息
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置链接器标志以允许共享库中的未定义符
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/ItcMultiDetect_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# set(LIB_ARCH "x86_64")
set(LIB_ARCH aarch64)
if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()
include_directories(${CMAKE_SOURCE_DIR})



# set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/GCC_10_4/opencv/opencv-linux-aarch64/share/OpenCV) #链接静态库
# set(OpenCV_DIR /home/firefly/ljh/opencv4.5-static/lib/cmake/opencv4)
# set(OpenCV_DIR /home/firefly/ljh/opencv4.5-dynamic/lib/cmake/opencv4)

# set(OpenCV_DIR /home/ljh/ljh_nfs/ItcMultiDetect/3rdparty/GCC_10_4/opencv/opencv-linux-aarch64/share/OpenCV)
set(OpenCV_DIR /home/firefly/ljh/opencv4.5-dynamic/lib/cmake/opencv4)
find_package(OpenCV REQUIRED) #在OpenCV_DIR路径中根据OpenCV包名寻找OpenCV库的配置文件


# 输出 OpenCV 配置
message(STATUS "OpenCV found in: ${OpenCV_DIR}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")


# set(CMAKE_INSTALL_RPATH "lib")

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src)


# 可执行文件
# add_executable(abnormal_detect
#     src/main_normal_det.cc
#     src/normal_det.cc
# )

add_library(abnormal_detect STATIC
          src/normal_det.cc
)

#为特定目标添加头文件搜索, 主工程add_subdirectory才能找到头文件
target_include_directories(abnormal_detect PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)


#将库链接到可执行文件               
target_link_libraries(abnormal_detect
  ${OpenCV_LIBS}
)


#链接phtead库
# find_package(Threads REQUIRED)
# target_link_libraries(Highthrow ${CMAKE_THREAD_LIBS_INIT})
