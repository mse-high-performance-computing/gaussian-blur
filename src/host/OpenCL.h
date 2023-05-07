//
// Created by kurbaniec on 07.05.2023.
//

#ifndef GAUSSIAN_BLUR_OPENCL_H
#define GAUSSIAN_BLUR_OPENCL_H

#include <string>
#include <optional>
#include <functional>
#include <CL/cl.h>
#include <memory>

namespace OpenCL {

    struct Argument {
        std::string key;
        cl_uint index;

        void* pointer;
        std::optional<std::function<void(void*)>> free;
        size_t size;
        cl_mem_flags flags;

        bool writeBuffer;
        cl_mem buffer;
    };



    struct App {
        cl_int status;
        cl_device_id device;
        cl_context context;
        cl_command_queue commandQueue;
        cl_program program;
        cl_kernel kernel;
        std::vector<std::shared_ptr<Argument>> arguments;
    };

    App setup();

    void createKernel(
        const std::string& filename,
        const std::string& kernel
    );

    std::shared_ptr<Argument> addArgument(
        App& app,
        std::string key,
        cl_uint index,
        void* pointer,
        std::optional<std::function<void(void*)>> free,
        size_t size,
        cl_mem_flags flags,
        bool writeBuffer
    );

    void checkDeviceCapabilities(
        std::function<bool(
            size_t maxWorkGroupSize,
            cl_uint maxWorkItemDimensions,
            size_t* maxWorkItemSizes
        )>
    );

    void enqueueKernel(
        cl_uint workDimensions,
        size_t* globalWorkSize
    );

    void readBuffer(
        std::shared_ptr<Argument>,
        cl_bool blockingRead
    );

    void release();






} // OpenCL

#endif //GAUSSIAN_BLUR_OPENCL_H
