//
// Created by kurbaniec on 07.05.2023.
//

#ifndef GAUSSIAN_BLUR_OPENCL_H
#define GAUSSIAN_BLUR_OPENCL_H

#include <string>
#include <optional>
#include <functional>
#if _WIN32
#include <CL/cl.h>
#elif __APPLE__
#include <OpenCL/opencl.h>
#endif
#include <memory>

namespace {
    std::string cl_errorstring(cl_int err);

    void checkStatus(cl_int err);

    void printCompilerError(cl_program program, cl_device_id device);
}

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

        Argument(std::string  key, cl_uint index, void* pointer,
                 const std::optional<std::function<void(void*)>>& free, size_t size, cl_mem_flags flags,
                 bool writeBuffer, cl_mem buffer);
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

    std::shared_ptr<Argument> addArgument(
        App& app,
        const std::string& key,
        cl_uint index,
        void* pointer,
        const std::optional<std::function<void(void*)>>& free,
        size_t size,
        cl_mem_flags flags,
        bool writeBuffer
    );

    void createKernel(
        App& app,
        const std::string& filename,
        const std::string& kernel
    );

    void checkDeviceCapabilities(
        App& app,
        std::function<bool(
            size_t maxWorkGroupSize,
            cl_uint maxWorkItemDimensions,
            size_t* maxWorkItemSizes
        )>
    );

    void enqueueKernel(
        App& app,
        cl_uint workDimensions,
        size_t* globalWorkSize
    );

    void readBuffer(
        App& app,
        std::shared_ptr<Argument> arg,
        cl_bool blockingRead
    );

    void release(App& app);






} // OpenCL

#endif //GAUSSIAN_BLUR_OPENCL_H
