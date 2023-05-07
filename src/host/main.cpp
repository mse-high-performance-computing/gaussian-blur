#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if _WIN32

#include <CL/cl.h>

#elif __APPLE__
#include <OpenCL/opencl.h>
#endif

#include <fstream>
#include <string>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include "OpenCL.h"


void printVector(int32_t* vector, unsigned int elementSize, const char* label) {
    printf("%s:\n", label);

    for (unsigned int i = 0; i < elementSize; ++i) {
        printf("%d ", vector[i]);
    }

    printf("\n");
}

struct Image {
    cl_int width;
    cl_int height;
    cl_int channels;
    size_t dataSize;
    cl_uchar* data;
};

// Image loadImage(const std::string& filename) {
//     int width, height, channels;
//     cl_uchar* imageInput = stbi_load(
//         "shuttle.png",
//         &width, &height, &channels, 0
//     );
//     if (imageInput == nullptr) {
//         printf("Error in loading the image\n");
//         exit(1);
//     }
//     if (channels != 3) {
//         printf("Unsupported channel size %i, only 3 is supported\n", channels);
//     }
//     printf(
//         "Loaded image with a width of %dpx, a height of %dpx and %d channels\n",
//         width, height, channels
//     );
//
//     size_t dataSize = width * height * channels * sizeof(cl_uchar);
//     auto* data = static_cast<cl_uchar *>(malloc(dataSize));
//     for (auto h = 0, index = 0; h < height; ++h) {
//         for (auto w = 0; w < width; ++w) {
//             data[index] = imageInput[index++];
//             data[index] = imageInput[index++];
//             data[index] = imageInput[index++];
//         }
//     }
//     stbi_image_free(imageInput);
//
//     return Image {
//         width, height, channels, dataSize, data
//     };
// }

int main(int argc, char** argv) {
    // input and output arrays
    /*const unsigned int elementSize = 10;
    size_t dataSize = elementSize * sizeof(int32_t);
    int32_t* vectorA = static_cast<int32_t*>(malloc(dataSize));
    int32_t* vectorB = static_cast<int32_t*>(malloc(dataSize));

    for (unsigned int i = 0; i < elementSize; ++i) {
        vectorA[i] = static_cast<int32_t>(i);
    }*/
    cl_int width, height, channels;
    cl_uchar* imageInput = stbi_load(
        "shuttle.png",
        &width, &height, &channels, 0
    );
    if (imageInput == nullptr) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf(
        "Loaded image with a width of %dpx, a height of %dpx and %d channels\n",
        width, height, channels
    );

    // See Moodle
/*
#define smooth_kernel_size 9
#define sigma 1.0

    int main()
    {
        // based on https://stackoverflow.com/questions/54614167/trying-to-implement-gaussian-filter-in-c

        double gauss[smooth_kernel_size][smooth_kernel_size];
        double sum = 0;
        int i, j;

        for (i = 0; i < smooth_kernel_size; i++) {
            for (j = 0; j < smooth_kernel_size; j++) {
                double x = i - (smooth_kernel_size - 1) / 2.0;
                double y = j - (smooth_kernel_size - 1) / 2.0;
                gauss[i][j] = 1.0 / (2.0 * M_PI * pow(sigma, 2.0)) * exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
                sum += gauss[i][j];
            }
        }

        for (i = 0; i < smooth_kernel_size; i++) {
            for (j = 0; j < smooth_kernel_size; j++) {
                gauss[i][j] /= sum;
            }
        }

        printf("2D Gaussian filter kernel:\n");
        for (i = 0; i < smooth_kernel_size; i++) {
            for (j = 0; j < smooth_kernel_size; j++) {
                printf("%f, ", gauss[i][j]);
            }
            printf("\n");
        }

        double gaussSeparated[smooth_kernel_size];

        for (i = 0; i < smooth_kernel_size; i++) {
            gaussSeparated[i] = sqrt(gauss[i][i]);
        }

        printf("1D Separated Gaussian filter kernel:\n");
        for (i = 0; i < smooth_kernel_size; i++) {
            printf("%f, ", gaussSeparated[i]);
        }
        printf("\n");

        return 0;
    }*/
    cl_int smoothKernelDimension = 3;
    size_t smoothKernelSize = sizeof(cl_float) * 9;
    auto* smoothKernel = static_cast<cl_float*>(malloc(smoothKernelSize));
    float smoothKernelInput[9] = {
        0.000134, 0.004432, 0.053991,
        0.241971, 0.398943, 0.241971,
        0.053991, 0.004432, 0.000134
    };
    std::copy(smoothKernelInput, smoothKernelInput + 9, smoothKernel);

    const unsigned int elementSize = width * height;
    size_t dataSize = width * height * channels * sizeof(cl_uchar);
    auto* imageOutput = static_cast<cl_uchar*>(malloc(dataSize));


    auto app = OpenCL::setup();

    /*// used for checking error status of api calls
    cl_int status;

    // retrieve the number of platforms
    cl_uint numPlatforms = 0;
    checkStatus(clGetPlatformIDs(0, NULL, &numPlatforms));

    if (numPlatforms == 0) {
        printf("Error: No OpenCL platform available!\n");
        exit(EXIT_FAILURE);
    }

    // select the platform
    cl_platform_id platform;
    checkStatus(clGetPlatformIDs(1, &platform, NULL));

    // retrieve the number of devices
    cl_uint numDevices = 0;
    checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));

    if (numDevices == 0) {
        printf("Error: No OpenCL device available for platform!\n");
        exit(EXIT_FAILURE);
    }

    // select the device
    cl_device_id device;
    checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

    // create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkStatus(status);

    // create command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
    checkStatus(status);

    // allocate two input and one output buffer for the three vectors
    cl_mem bufferImageInput = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
    checkStatus(status);
    cl_mem bufferImageOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);
    checkStatus(status);
    cl_mem bufferWidth = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, &status);
    checkStatus(status);
    cl_mem bufferHeight = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, &status);
    checkStatus(status);
    cl_mem bufferSmoothKernel = clCreateBuffer(context, CL_MEM_READ_ONLY, smoothKernelSize, NULL, &status);
    checkStatus(status);
    cl_mem bufferSmoothKernelDimension = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, &status);
    checkStatus(status);*/

    OpenCL::addArgument(
        app, "imageInput", 0, imageInput,
        [](void* pointer) { stbi_image_free(pointer); },
        dataSize, CL_MEM_READ_ONLY, true
    );
    printf("1");
    auto outArg = OpenCL::addArgument(
        app, "imageOutput", 1, imageOutput,
        [](void* pointer) { free(pointer); },
        dataSize, CL_MEM_WRITE_ONLY, false
    );
    printf("2");
    OpenCL::addArgument(
        app, "width", 2, &width, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    printf("3");
    OpenCL::addArgument(
        app, "height", 3, &height, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    printf("4");
    OpenCL::addArgument(
        app, "smoothKernel", 4, smoothKernel,
        [](void* pointer) { free(pointer); },
        smoothKernelSize, CL_MEM_READ_ONLY, true
    );
    printf("5");
    OpenCL::addArgument(
        app, "smoothKernelDimension", 5, &smoothKernelDimension, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    printf("6");


    /*// write data from the input vectors to the buffers
    checkStatus(clEnqueueWriteBuffer(commandQueue, bufferImageInput, CL_TRUE, 0, dataSize, imageInput, 0, NULL, NULL));
    checkStatus(clEnqueueWriteBuffer(commandQueue, bufferWidth, CL_TRUE, 0, sizeof(cl_int), &width, 0, NULL, NULL));
    checkStatus(clEnqueueWriteBuffer(commandQueue, bufferHeight, CL_TRUE, 0, sizeof(cl_int), &height, 0, NULL, NULL));
    printf("0");
    checkStatus(clEnqueueWriteBuffer(commandQueue, bufferSmoothKernel, CL_TRUE, 0, smoothKernelSize, smoothKernel, 0, NULL, NULL));
    printf("1");
    checkStatus(clEnqueueWriteBuffer(commandQueue, bufferSmoothKernelDimension, CL_TRUE, 0, sizeof(cl_int), &smoothKernelDimension, 0, NULL, NULL));

    printf("2");*/

    OpenCL::createKernel(app, "kernel/vector_add.cl", "gaussian_blur");
    /* // read the kernel source
     const char* kernelFileName = "kernel/vector_add.cl";
     std::ifstream ifs(kernelFileName);
     if (!ifs.good()) {
         printf("Error: Could not open kernel with file name %s!\n", kernelFileName);
         exit(EXIT_FAILURE);
     }

     std::string programSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
     const char* programSourceArray = programSource.c_str();
     size_t programSize = programSource.length();

     // create the program
     cl_program program = clCreateProgramWithSource(context, 1, static_cast<const char**>(&programSourceArray),
                                                    &programSize, &status);
     checkStatus(status);

     // build the program
     status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
     if (status != CL_SUCCESS) {
         printCompilerError(program, device);
         exit(EXIT_FAILURE);
     }

     // create the vector addition kernel
     cl_kernel kernel = clCreateKernel(program, "gaussian_blur", &status);
     checkStatus(status);

     // set the kernel arguments
     checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferImageInput));
     checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferImageOutput));
     checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferWidth));
     checkStatus(clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferHeight));
     checkStatus(clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferSmoothKernel));
     checkStatus(clSetKernelArg(kernel, 5, sizeof(cl_mem), &bufferSmoothKernelDimension));
 */

    // TODO
    /*// output device capabilities
    size_t maxWorkGroupSize;
    checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL));
    printf("Device Capabilities: Max work items in single group: %zu\n", maxWorkGroupSize);

    cl_uint maxWorkItemDimensions;
    checkStatus(
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, NULL));
    printf("Device Capabilities: Max work item dimensions: %u\n", maxWorkItemDimensions);

    size_t* maxWorkItemSizes = static_cast<size_t*>(malloc(maxWorkItemDimensions * sizeof(size_t)));
    checkStatus(
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimensions * sizeof(size_t), maxWorkItemSizes,
                        NULL));
    printf("Device Capabilities: Max work items in group per dimension:");
    for (cl_uint i = 0; i < maxWorkItemDimensions; ++i)
        printf(" %u:%zu", i, maxWorkItemSizes[i]);
    printf("\n");
    free(maxWorkItemSizes);*/

    // execute the kernel
    // ndrange capabilites only need to be checked when we specify a local work group size manually
    // in our case we provide NULL as local work group size, which means groups get formed automatically

    std::cout << "width " << width << ", height " << height;

    size_t globalWorkSize[2] = {static_cast<size_t>(width),
                                static_cast<size_t>(height)}; // https://stackoverflow.com/a/31379085
    OpenCL::enqueueKernel(app, 2, globalWorkSize);
    // checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL));

    // read the device output buffer to the host output array
    // `clEnqueueReadBuffer` does not wait for the kernel unless `blocking_read` is set to `CL_TRUE`
    // checkStatus(clEnqueueReadBuffer(commandQueue, bufferImageOutput, CL_TRUE, 0, dataSize, imageOutput, 0, NULL, NULL));
    OpenCL::readBuffer(app, outArg, CL_TRUE);

    // output result
    /*printVector(vectorA, elementSize, "Input A");
    printVector(vectorB, elementSize, "Input B");*/
    stbi_write_png(
        "blurred.png", width, height,
        channels, imageOutput, width * channels
    );

    // release allocated resources
    OpenCL::release(app);
    /*free(vectorB);
    free(vectorA);*/

    /*stbi_image_free(imageInput);
    free(imageOutput);
    free(smoothKernel);

    // release opencl objects
    checkStatus(clReleaseKernel(kernel));
    checkStatus(clReleaseProgram(program));
    checkStatus(clReleaseMemObject(bufferImageInput));
    checkStatus(clReleaseMemObject(bufferHeight));
    checkStatus(clReleaseMemObject(bufferWidth));
    checkStatus(clReleaseMemObject(bufferImageOutput));
    checkStatus(clReleaseMemObject(bufferSmoothKernel));
    checkStatus(clReleaseMemObject(bufferSmoothKernelDimension));

    checkStatus(clReleaseCommandQueue(commandQueue));
    checkStatus(clReleaseContext(context));*/

    exit(EXIT_SUCCESS);
}
