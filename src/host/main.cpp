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
    printf("gaussian-blur\n");
    std::vector<std::string> args(&argv[0], &argv[0 + argc]);
    args.erase(args.begin());

    auto argsCount = args.size();


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

    cl_int smoothKernelDimension = 3;
    size_t smoothKernelSize = sizeof(cl_float) * 9;
    // auto* smoothKernel = static_cast<cl_float*>(malloc(smoothKernelSize));

    // Initialised on stack
    float smoothKernel[9] = {
        0.000134, 0.004432, 0.053991,
        0.241971, 0.398943, 0.241971,
        0.053991, 0.004432, 0.000134
    };
    // std::copy(smoothKernelInput, smoothKernelInput + 9, smoothKernel);

    const unsigned int elementSize = width * height;
    size_t dataSize = width * height * channels * sizeof(cl_uchar);
    auto* imageOutput = static_cast<cl_uchar*>(malloc(dataSize));


    // select the platform
    // retrieve the number of devices
    // select the device
    // create context
    // create command queue
    auto app = OpenCL::setup();

    // allocate buffers
    OpenCL::addArgument(
        app, "imageInput", 0, imageInput,
        [](void* pointer) { stbi_image_free(pointer); },
        dataSize, CL_MEM_READ_ONLY, true
    );
    auto outArg = OpenCL::addArgument(
        app, "imageOutput", 1, imageOutput,
        [](void* pointer) { free(pointer); },
        dataSize, CL_MEM_WRITE_ONLY, false
    );
    OpenCL::addArgument(
        app, "width", 2, &width, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        app, "height", 3, &height, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        app, "smoothKernel", 4, smoothKernel,std::nullopt,
        smoothKernelSize, CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        app, "smoothKernelDimension", 5, &smoothKernelDimension, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );

    // read the kernel source
    // create the program
    // build the program
    // create the given kernel
    // set the kernel arguments
    OpenCL::createKernel(app, "kernel/vector_add.cl", "gaussian_blur");

    // check device capabilities
    // check if image fits
    OpenCL::checkDeviceCapabilities(app, [width, height](auto maxWorkGroupSize, auto maxWorkItemDimensions, auto* maxWorkItemSizes) {
        if (maxWorkItemDimensions < 2) return false;
        if (maxWorkItemSizes[0] < width) return false;
        if (maxWorkItemSizes[1] < height) return false;
        return true;
    });

    // execute the kernel
    // ndrange capabilites only need to be checked when we specify a local work group size manually
    // in our case we provide NULL as local work group size, which means groups get formed automatically
    size_t globalWorkSize[2] = {static_cast<size_t>(width),
                                static_cast<size_t>(height)}; // https://stackoverflow.com/a/31379085
    OpenCL::enqueueKernel(app, 2, globalWorkSize);

    // read the device output buffer to the host output array
    OpenCL::readBuffer(app, outArg, CL_TRUE);

    // output result to file
    stbi_write_png(
        "blurred.png", width, height,
        channels, imageOutput, width * channels
    );

    // release allocated resources
    OpenCL::release(app);

    exit(EXIT_SUCCESS);
}
