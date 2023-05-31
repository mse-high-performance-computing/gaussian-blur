//
// Created by kurbaniec on 07.05.2023.
//

#ifndef GAUSSIAN_BLUR_OPENCL_H
#define GAUSSIAN_BLUR_OPENCL_H


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#if _WIN32
#include <CL/cl.h>
#elif __APPLE__
#include <OpenCL/opencl.h>
#endif

#include <string>
#include <optional>
#include <functional>
#include <memory>
#include <map>

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

        void freeResources();
    };

    struct App {
        cl_int status;
        cl_device_id device;
        cl_context context;
        cl_command_queue commandQueue;
        cl_program program;
        cl_kernel kernel;
        std::map<cl_uint, std::shared_ptr<Argument>> arguments;
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

    std::shared_ptr<Argument> addLocalArgument(
        App& app,
        const std::string& key,
        cl_uint index,
        size_t size
    );

    void removeArgument(App& app, const std::shared_ptr<Argument>& arg);

    void changeArgumentIndex(App& app, const std::shared_ptr<Argument>& arg, cl_uint index);

    void createKernel(
        App& app,
        const std::string& filename,
        const std::string& kernel
    );

    void refreshKernelArguments(App& app);

    void checkDeviceCapabilities(
        App& app,
        const std::function<bool(
            size_t maxWorkGroupSize,
            cl_uint maxWorkItemDimensions,
            size_t* maxWorkItemSizes,
            cl_ulong maxLocalMemory
        )>& check
    );

    void enqueueKernel(
        App& app,
        cl_uint workDimensions,
        size_t* globalWorkSize,
        size_t* localWorkSize,
        cl_uint num_events_in_wait_list,
        cl_event* event_wait,
        cl_event* event
    );

    void waitForEvents(cl_uint numEvents, const cl_event* eventList);

    void readBuffer(
        App& app,
        const std::shared_ptr<Argument>& arg,
        cl_bool blockingRead
    );

    void release(App& app);






} // OpenCL

#endif //GAUSSIAN_BLUR_OPENCL_H
