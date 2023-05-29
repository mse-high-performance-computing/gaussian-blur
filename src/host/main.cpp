

#include <string>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include "OpenCL.h"


struct Image {
    cl_int width;
    cl_int height;
    cl_int channels;
    size_t size;
    cl_uchar* data;
};

Image loadImage(const std::string& filename) {
    int width, height, channels;
    cl_uchar* data = stbi_load(
        filename.c_str(),
        &width, &height, &channels, 0
    );
    if (data == nullptr) {
        printf("Error in loading the image\n");
        exit(1);
    }
    if (channels != 3) {
        printf("Unsupported channel size %i, only 3 is supported\n", channels);
        exit(1);
    }
    printf(
        "Loaded image with a width of %dpx, a height of %dpx and %d channels\n",
        width, height, channels
    );

    size_t size = width * height * channels * sizeof(cl_uchar);

    return Image {
        width, height, channels, size, data
    };
}

struct SmoothKernel {
    cl_int dimension;
    size_t size;
    cl_float * data;
};

void removeChar(std::string& str, char c) {
    str.erase(std::remove(str.begin(), str.end(), c), str.end());
}

std::vector<std::string> splitStr(std::string& str, char delimiter) {
    std::vector<std::string> values;
    std::stringstream stream(str);
    std::string tmp;
    while (std::getline(stream, tmp, delimiter))
        values.push_back(tmp);
    return values;
}

cl_float* strToFloat(std::vector<std::string>& strings) {
    auto* values = static_cast<cl_float*>(malloc(sizeof(cl_float) * strings.size()));
    for (int i = 0; i < strings.size(); ++i) {
        values[i] = std::stof(strings[i]);
    }
    return values;
}

SmoothKernel loadSmoothKernel(const std::string& kernelInput) {
    auto kernelRaw = kernelInput;
    removeChar(kernelRaw, '(');
    removeChar(kernelRaw, ')');
    auto kernelSplit = splitStr(kernelRaw, ',');
    // Check if dimension is ok (odd + perfect square)
    auto length = kernelSplit.size();
    long long dimension = std::floor(sqrt(length));
    if (dimension * dimension == length && length && length % 2 != 0) {
        auto data = strToFloat(kernelSplit);
        auto size = length * sizeof(cl_float);
        return SmoothKernel {static_cast<cl_int>((cl_uint)dimension), size, data };
    } else {
        printf("Unsupported kernel size: %zu\n", length);
        exit(1);
    }
}

int main(int argc, char** argv) {
    printf("gaussian-blur\n");
    std::vector<std::string> args(&argv[0], &argv[0 + argc]);
    args.erase(args.begin());

    auto argsCount = args.size();
    std::string filename;
    std::string kernelInput;

    if (argsCount == 1) {
        filename = args[0];
        kernelInput = "(0.000134,0.004432,0.053991,"
                      "0.241971,0.398943,0.241971,"
                      "0.053991,0.004432,0.000134)";
    } else if (argsCount == 2) {
        filename = args[0];
        kernelInput = args[1];
    } else {
        printf("Invalid input\n");
        printf("Usage [filename] [optional: kernel]\n");
        exit(EXIT_FAILURE);
    }

    printf("Parameters:\n");
    printf("  File: %s\n", filename.c_str());
    printf("  Kernel: %s\n", kernelInput.c_str());

    auto imageInput = loadImage(filename);
    auto* tmpImage = static_cast<cl_uchar*>(malloc(imageInput.size));
    auto smoothKernel = loadSmoothKernel(kernelInput);

    // select the platform
    // retrieve the number of devices
    // select the device
    // create context
    // create command queue
    auto appHorizontal = OpenCL::setup();
    cl_bool isHorizontal = true;

    // allocate buffers
    OpenCL::addArgument(
        appHorizontal, "imageInput", 0, imageInput.data,
        [](void* pointer) { stbi_image_free(pointer); },
        imageInput.size, CL_MEM_READ_ONLY, true
    );
    auto tmpOutput = OpenCL::addArgument(
        appHorizontal, "imageOutput", 1, tmpImage,
        [](void* pointer) { free(pointer); },
        imageInput.size, CL_MEM_WRITE_ONLY, false
    );
    OpenCL::addArgument(
        appHorizontal, "width", 2, &imageInput.width, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        appHorizontal, "height", 3, &imageInput.height, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        appHorizontal, "smoothKernel", 4, smoothKernel.data, std::nullopt,
        smoothKernel.size, CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        appHorizontal, "smoothKernelDimension", 5, &smoothKernel.dimension, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        appHorizontal, "horizontal", 6, &isHorizontal, std::nullopt,
        sizeof(cl_bool), CL_MEM_READ_ONLY, true
    );
    
    
    // read the kernel source
    // create the program
    // build the program
    // create the given kernel
    // set the kernel arguments
    OpenCL::createKernel(appHorizontal, "kernel/gaussian_blur.cl", "gaussian_blur");

    // check device capabilities
    // check if image fits
    size_t width = imageInput.width;
    size_t height = imageInput.height;
    OpenCL::checkDeviceCapabilities(appHorizontal, [width, height](auto maxWorkGroupSize, auto maxWorkItemDimensions, auto* maxWorkItemSizes) {
        if (maxWorkItemDimensions < 2) return false;
        if (maxWorkItemSizes[0] < width) return false;
        if (maxWorkItemSizes[1] < height) return false;
        return true;
    });
    
    // Create events for synchronization
    cl_event horizontalEvent;
    cl_event verticalEvent;

    // execute the kernel
    // ndrange capabilites only need to be checked when we specify a local work group size manually
    // in our case we provide NULL as local work group size, which means groups get formed automatically
    size_t globalWorkSize[2] = {width, height}; // https://stackoverflow.com/a/31379085
    size_t localWorkSizeHorizontal[2] = { 1, height };
    
    OpenCL::enqueueKernel(appHorizontal, 2, globalWorkSize,localWorkSizeHorizontal,0,nullptr,&horizontalEvent);
    
    clWaitForEvents(1, &horizontalEvent);


    // read the device output buffer to the host output array
    OpenCL::readBuffer(appHorizontal, tmpOutput, CL_TRUE);

    auto* outputImage = static_cast<cl_uchar*>(malloc(imageInput.size));

    // select the platform
    // retrieve the number of devices
    // select the device
    // create context
    // create command queue
    auto appVertical = OpenCL::setup();
    cl_bool isVertical = true;

    // allocate buffers
    OpenCL::addArgument(
        appVertical, "imageInput", 0, tmpImage,
        [](void* pointer) { stbi_image_free(pointer); },
        imageInput.size, CL_MEM_READ_ONLY, true
    );
    auto output = OpenCL::addArgument(
        appVertical, "imageOutput", 1, outputImage,
        [](void* pointer) { free(pointer); },
        imageInput.size, CL_MEM_WRITE_ONLY, false
    );
    OpenCL::addArgument(
        appVertical, "width", 2, &imageInput.width, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        appVertical, "height", 3, &imageInput.height, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        appVertical, "smoothKernel", 4, smoothKernel.data, std::nullopt,
        smoothKernel.size, CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        appVertical, "smoothKernelDimension", 5, &smoothKernel.dimension, std::nullopt,
        sizeof(cl_int), CL_MEM_READ_ONLY, true
    );
    OpenCL::addArgument(
        appVertical, "horizontal", 6, &isVertical, std::nullopt,
        sizeof(cl_bool), CL_MEM_READ_ONLY, true
    );
    
    
    // read the kernel source
    // create the program
    // build the program
    // create the given kernel
    // set the kernel arguments
    OpenCL::createKernel(appVertical, "kernel/gaussian_blur.cl", "gaussian_blur");
    
    size_t localWorkSizeVertical[2] = { width, 1 };
    
    OpenCL::enqueueKernel(appVertical, 2, globalWorkSize,localWorkSizeVertical,0,nullptr,&verticalEvent);
    
    clWaitForEvents(1, &verticalEvent);


    // read the device output buffer to the host output array
    OpenCL::readBuffer(appVertical, output, CL_TRUE);
    
    // output result to file
    stbi_write_png(
        "blurred.png", imageInput.width, imageInput.height,
        imageInput.channels, outputImage, imageInput.width * imageInput.channels
    );
    printf("Blurred image written in 'blurred.png'\n");

    // release allocated resources
    OpenCL::release(appHorizontal);

    exit(EXIT_SUCCESS);
}
