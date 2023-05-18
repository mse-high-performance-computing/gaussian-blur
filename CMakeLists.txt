cmake_minimum_required(VERSION 3.24)
project(gaussian-blur)

set(CMAKE_CXX_STANDARD 17)

# When encountering linking errors with msvc in CLion please refer to:
# https://youtrack.jetbrains.com/issue/CPP-26650/MSVS2019-toolchain-linking-fails-with-cmd.exe-is-not-recognized-as-an-internal-or-external-command-if-PATH-is-longer-than-2047
# message("Current PATH is $ENV{PATH}")

# Find OpenCL library
# https://cmake.org/cmake/help/latest/module/FindOpenCL.html
find_package(OpenCL REQUIRED)
# One can use the following import instead of `find_package(OpenCL REQUIRED)` to force Nvidia's OpenCL library
# Can be useful when `find_package(OpenCL REQUIRED)` find and uses another implementation (eg. Intel)
# https://stackoverflow.com/a/67642989
# find_package(CUDAToolkit 12.1 REQUIRED)

add_executable(gaussian-blur src/host/main.cpp src/host/OpenCL.h src/host/OpenCL.cpp)
target_link_libraries(gaussian-blur PRIVATE OpenCL::OpenCL)

# STB
include_directories(${PROJECT_SOURCE_DIR}/dependencies/)

# Copy kernels to output directory
set(KERNEL_DIR ${PROJECT_SOURCE_DIR}/src/kernel)
add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${KERNEL_DIR} ${CMAKE_BINARY_DIR}/kernel)

# Copy images to output directory
set(IMAGE_DIR ${PROJECT_SOURCE_DIR}/images)
add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${IMAGE_DIR} ${CMAKE_BINARY_DIR}/images)