//
// Created by kurbaniec on 07.05.2023.
//

#include "OpenCL.h"

#include <utility>
#include <fstream>

namespace OpenCL {

    Argument::Argument(std::string key, cl_uint index, void* pointer,
                       const std::optional<std::function<void(void*)>>& free, size_t size, cl_mem_flags flags,
                       bool writeBuffer, cl_mem buffer) : key(std::move(key)), index(index), pointer(pointer),
                                                          free(free),
                                                          size(size), flags(flags), writeBuffer(writeBuffer),
                                                          buffer(buffer) {}

    void Argument::freeResources() {
        if (buffer == nullptr) return; // local memory

        checkStatus(clReleaseMemObject(buffer));
        if (auto freeFn = free) {
            (*freeFn)(pointer);
        }
    }

    App setup() {
        cl_int status;
        // retrieve the number of platforms
        cl_uint numPlatforms = 0;
        checkStatus(clGetPlatformIDs(0, nullptr, &numPlatforms));

        if (numPlatforms == 0) {
            printf("Error: No OpenCL platform available!\n");
            exit(EXIT_FAILURE);
        }

        // select the platform
        cl_platform_id platform;
        checkStatus(clGetPlatformIDs(1, &platform, nullptr));

        // retrieve the number of devices
        cl_uint numDevices = 0;
        checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices));

        if (numDevices == 0) {
            printf("Error: No OpenCL device available for platform!\n");
            exit(EXIT_FAILURE);
        }

        // select the device
        cl_device_id device;
        checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr));

        // create context
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        checkStatus(status);

        // create command queue
        cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
        checkStatus(status);

        return App{
            status, device, context, commandQueue,
            nullptr, nullptr,
            std::map<cl_uint, std::shared_ptr<Argument>>()
        };
    }

    std::shared_ptr<Argument>
    addArgument(
        App& app,
        const std::string& key, cl_uint index, void* pointer, const std::optional<std::function<void(void*)>>& free,
        size_t size, cl_mem_flags flags, bool writeBuffer
    ) {
        if (app.arguments.count(index)) {
            printf("Error: Argument %i already present", index);
            throw std::runtime_error("Argument " + std::to_string(index) + " already present");
        }

        cl_mem buffer = clCreateBuffer(app.context, flags, size, nullptr, &app.status);
        checkStatus(app.status);

        // write data from the input to the buffers
        if (writeBuffer)
            checkStatus(clEnqueueWriteBuffer(
                app.commandQueue, buffer,
                CL_TRUE, 0, size, pointer,
                0, nullptr, nullptr
            ));

        auto arg = std::make_shared<Argument>(key, index, pointer, free, size, flags, writeBuffer, buffer);
        app.arguments.insert({index, arg});

        return arg;
    }

    std::shared_ptr<Argument> addLocalArgument(
        App& app,
        const std::string& key,
        cl_uint index,
        size_t size
    ) {
        if (app.arguments.count(index)) {
            printf("Error: Argument %i already present", index);
            throw std::runtime_error("Argument " + std::to_string(index) + " already present");
        }

        auto arg = std::make_shared<Argument>(key, index, nullptr, free, size, CL_MEM_FLAGS, false, nullptr);
        app.arguments.insert({index, arg});

        return arg;
    }

    void removeArgument(App& app, const std::shared_ptr<Argument>& arg) {
        // Free resources
        arg->freeResources();
        app.arguments.erase(arg->index);
    }

    void changeArgumentIndex(App& app, const std::shared_ptr<Argument>& arg, cl_uint index) {
        if (app.arguments.count(index)) {
            printf("Error: Argument %i already present", index);
            throw std::runtime_error("Argument " + std::to_string(index) + " already present");
        }
        app.arguments.erase(arg->index);
        arg->index = index;
        app.arguments.insert({index, arg});
    }

    void createKernel(App& app, const std::string& filename, const std::string& kernel) {
        // read the kernel source
        std::ifstream ifs(filename);
        if (!ifs.good()) {
            printf("Error: Could not open kernel with file name %s!\n", filename.c_str());
            exit(EXIT_FAILURE);
        }

        std::string programSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        const char* programSourceArray = programSource.c_str();
        size_t programSize = programSource.length();

        // create the program
        app.program = clCreateProgramWithSource(app.context, 1, static_cast<const char**>(&programSourceArray),
                                                &programSize, &app.status);
        checkStatus(app.status);

        // build the program
        app.status = clBuildProgram(app.program, 1, &app.device, nullptr, nullptr, nullptr);
        if (app.status != CL_SUCCESS) {
            printCompilerError(app.program, app.device);
            exit(EXIT_FAILURE);
        }

        // create the given kernel
        app.kernel = clCreateKernel(app.program, kernel.c_str(), &app.status);
        checkStatus(app.status);

        // set the kernel arguments
        refreshKernelArguments(app);
    }

    void refreshKernelArguments(App& app) {
        for (auto& [_, arg]: app.arguments) {
            // Differentiate between global & local (buffer=nullptr) memory arguments
            auto argSize = arg->buffer == nullptr ? arg->size : sizeof(cl_mem);
            auto argValue = arg->buffer == nullptr ? nullptr : &arg->buffer;
            checkStatus(clSetKernelArg(
                app.kernel, arg->index, argSize, argValue
            ));
        }
    }

    void checkDeviceCapabilities(
        App& app,
        const std::function<bool(
            size_t maxWorkGroupSize,
            cl_uint maxWorkItemDimensions,
            size_t* maxWorkItemSizes,
            cl_ulong maxLocalMemory
        )>& check
    ) {
        // output device capabilities
        size_t maxWorkGroupSize;
        checkStatus(clGetDeviceInfo(
            app.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
            &maxWorkGroupSize, nullptr
        ));
        printf("Device Capabilities: Max work items in single group: %zu\n", maxWorkGroupSize);

        cl_uint maxWorkItemDimensions;
        checkStatus(clGetDeviceInfo(
            app.device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint),
            &maxWorkItemDimensions, nullptr
        ));
        printf("Device Capabilities: Max work item dimensions: %u\n", maxWorkItemDimensions);

        auto* maxWorkItemSizes = static_cast<size_t*>(malloc(maxWorkItemDimensions * sizeof(size_t)));
        checkStatus(clGetDeviceInfo(
            app.device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimensions * sizeof(size_t),
            maxWorkItemSizes, nullptr
        ));
        printf("Device Capabilities: Max work items in group per dimension:");
        for (cl_uint i = 0; i < maxWorkItemDimensions; ++i)
            printf(" %u:%zu", i, maxWorkItemSizes[i]);
        printf("\n");

        cl_ulong maxLocalMemory;
        clGetDeviceInfo(app.device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &maxLocalMemory, 0);
        printf("Device Capabilities: Max local memory: %llu\n", maxLocalMemory);

        auto ok = check(maxWorkGroupSize, maxWorkItemDimensions, maxWorkItemSizes, maxLocalMemory);
        free(maxWorkItemSizes);

        if (!ok) {
            printf("Capability Error: Check returned false\n");
            exit(EXIT_FAILURE);
        }
    }

    void enqueueKernel(App& app, cl_uint workDimensions, size_t* globalWorkSize, size_t* localWorkSize,
                       cl_uint num_events_in_wait_list, cl_event* event_wait, cl_event* event) {
        // execute the kernel
        // ndrange capabilites only need to be checked when we specify a local work group size manually
        // in our case we provide NULL as local work group size, which means groups get formed automatically
        checkStatus(clEnqueueNDRangeKernel(
            app.commandQueue, app.kernel, workDimensions,
            nullptr, globalWorkSize, localWorkSize,
            num_events_in_wait_list, event_wait, event
        ));
    }

    void waitForEvents(cl_uint numEvents, const cl_event* eventList) {
        clWaitForEvents(numEvents, eventList);
    }

    void readBuffer(App& app, const std::shared_ptr<Argument>& arg, cl_bool blockingRead) {
        // read the device output buffer to the host output array
        // `clEnqueueReadBuffer` does not wait for the kernel unless `blocking_read` is set to `CL_TRUE`
        checkStatus(clEnqueueReadBuffer(
            app.commandQueue, arg->buffer, blockingRead,
            0, arg->size, arg->pointer, 0, nullptr, nullptr
        ));
    }

    void release(App& app) {
        // release allocated resources
        checkStatus(clReleaseKernel(app.kernel));
        checkStatus(clReleaseProgram(app.program));

        for (auto& [_, arg]: app.arguments) {
            arg->freeResources();
        }

        checkStatus(clReleaseCommandQueue(app.commandQueue));
        checkStatus(clReleaseContext(app.context));
    }


} // OpenCL

namespace {
    std::string cl_errorstring(cl_int err) {
        switch (err) {
            case CL_SUCCESS:
                return {"Success"};
            case CL_DEVICE_NOT_FOUND:
                return {"Device not found"};
            case CL_DEVICE_NOT_AVAILABLE:
                return {"Device not available"};
            case CL_COMPILER_NOT_AVAILABLE:
                return {"Compiler not available"};
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                return {"Memory object allocation failure"};
            case CL_OUT_OF_RESOURCES:
                return {"Out of resources"};
            case CL_OUT_OF_HOST_MEMORY:
                return {"Out of host memory"};
            case CL_PROFILING_INFO_NOT_AVAILABLE:
                return {"Profiling information not available"};
            case CL_MEM_COPY_OVERLAP:
                return {"Memory copy overlap"};
            case CL_IMAGE_FORMAT_MISMATCH:
                return {"Image format mismatch"};
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:
                return {"Image format not supported"};
            case CL_BUILD_PROGRAM_FAILURE:
                return {"Program build failure"};
            case CL_MAP_FAILURE:
                return {"Map failure"};
            case CL_MISALIGNED_SUB_BUFFER_OFFSET:
                return {"Misaligned sub buffer offset"};
            case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
                return {"Exec status error for events in wait list"};
            case CL_INVALID_VALUE:
                return {"Invalid value"};
            case CL_INVALID_DEVICE_TYPE:
                return {"Invalid device type"};
            case CL_INVALID_PLATFORM:
                return {"Invalid platform"};
            case CL_INVALID_DEVICE:
                return {"Invalid device"};
            case CL_INVALID_CONTEXT:
                return {"Invalid context"};
            case CL_INVALID_QUEUE_PROPERTIES:
                return {"Invalid queue properties"};
            case CL_INVALID_COMMAND_QUEUE:
                return {"Invalid command queue"};
            case CL_INVALID_HOST_PTR:
                return {"Invalid host pointer"};
            case CL_INVALID_MEM_OBJECT:
                return {"Invalid memory object"};
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
                return {"Invalid image format descriptor"};
            case CL_INVALID_IMAGE_SIZE:
                return {"Invalid image size"};
            case CL_INVALID_SAMPLER:
                return {"Invalid sampler"};
            case CL_INVALID_BINARY:
                return {"Invalid binary"};
            case CL_INVALID_BUILD_OPTIONS:
                return {"Invalid build options"};
            case CL_INVALID_PROGRAM:
                return {"Invalid program"};
            case CL_INVALID_PROGRAM_EXECUTABLE:
                return {"Invalid program executable"};
            case CL_INVALID_KERNEL_NAME:
                return {"Invalid kernel name"};
            case CL_INVALID_KERNEL_DEFINITION:
                return {"Invalid kernel definition"};
            case CL_INVALID_KERNEL:
                return {"Invalid kernel"};
            case CL_INVALID_ARG_INDEX:
                return {"Invalid argument index"};
            case CL_INVALID_ARG_VALUE:
                return {"Invalid argument value"};
            case CL_INVALID_ARG_SIZE:
                return {"Invalid argument size"};
            case CL_INVALID_KERNEL_ARGS:
                return {"Invalid kernel arguments"};
            case CL_INVALID_WORK_DIMENSION:
                return {"Invalid work dimension"};
            case CL_INVALID_WORK_GROUP_SIZE:
                return {"Invalid work group size"};
            case CL_INVALID_WORK_ITEM_SIZE:
                return {"Invalid work item size"};
            case CL_INVALID_GLOBAL_OFFSET:
                return {"Invalid global offset"};
            case CL_INVALID_EVENT_WAIT_LIST:
                return {"Invalid event wait list"};
            case CL_INVALID_EVENT:
                return {"Invalid event"};
            case CL_INVALID_OPERATION:
                return {"Invalid operation"};
            case CL_INVALID_GL_OBJECT:
                return {"Invalid OpenGL object"};
            case CL_INVALID_BUFFER_SIZE:
                return {"Invalid buffer size"};
            case CL_INVALID_MIP_LEVEL:
                return {"Invalid mip-map level"};
            case CL_INVALID_GLOBAL_WORK_SIZE:
                return {"Invalid gloal work size"};
            case CL_INVALID_PROPERTY:
                return {"Invalid property"};
            default:
                return {"Unknown error code"};
        }
    }

    void checkStatus(cl_int err) {
        if (err != CL_SUCCESS) {
            printf("OpenCL Error: %s \n", cl_errorstring(err).c_str());
            exit(EXIT_FAILURE);
        }
    }

    void printCompilerError(cl_program program, cl_device_id device) {
        cl_int status;
        size_t logSize;
        char* log;

        // get log size
        status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        checkStatus(status);

        // allocate space for log
        log = static_cast<char*>(malloc(logSize));
        if (!log) {
            exit(EXIT_FAILURE);
        }

        // read the log
        status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        checkStatus(status);

        // print the log
        printf("Build Error: %s\n", log);
    }
}
