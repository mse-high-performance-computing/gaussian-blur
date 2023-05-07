cmake_minimum_required(VERSION 3.24)
project(gaussian-blur)

set(CMAKE_CXX_STANDARD 17)

# https://youtrack.jetbrains.com/issue/CPP-26650/MSVS2019-toolchain-linking-fails-with-cmd.exe-is-not-recognized-as-an-internal-or-external-command-if-PATH-is-longer-than-2047
# message("Current PATH is $ENV{PATH}")

# https://stackoverflow.com/a/67642989
find_package(CUDAToolkit 12.1 REQUIRED)

add_executable(gaussian-blur src/host/main.cpp src/host/OpenCL.h src/host/OpenCL.cpp)
target_link_libraries(gaussian-blur PRIVATE CUDA::OpenCL)

# STB
include_directories(${PROJECT_SOURCE_DIR}/dependencies/)

# Copy kernels to output directory
set(KERNEL_DIR ${PROJECT_SOURCE_DIR}/src/kernel)
add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${KERNEL_DIR} ${CMAKE_BINARY_DIR}/kernel)