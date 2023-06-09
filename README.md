<div align="center">
  <p>Julian Mucherl | Kacper Urbaniec | Max Zojer</p>
  <h1><ins>gaussian-blur</ins></h1>
</div>

## 🛠️ Build

```bash
mkdir cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release -B cmake-build-release/ .
cmake --build cmake-build-release/ --target gaussian-blur -- -j 9
```

When using Visual Studio, please refer to the [official Microsoft documentation on importing CMake projects](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-170).

## 🚀 Run

```bash
cd cmake-build-release/
.\gaussian-blur.exe [filename] [optional: kernel]
# .\gaussian-blur.exe "images/shuttle.png" "(0.000134,0.004432,0.053991,0.241971,0.398943,0.241971,0.053991,0.004432,0.000134)"
```

> Note: The kernel argument must be odd, as asymmetric kernels are not desirable for gaussian blur.

### Create Kernel

One can generate Gaussian filter kernels (2D filter for simple and separable filter for
optimized version) using the generator program found in the following. Kernel size and blur strength is
adjusted by changing the appropriate defines. To run the program, one can simply copy-paste it into
an online Cpp Debugger (e.g.: https://www.onlinegdb.com).

Use the output of the *1D Separated Gaussian filter kernel* as kernel input for the gaussian blur application.

```c++
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

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
}
```



## Sources

* Slide decks
* https://github.com/nothings/stb
* http://chanhaeng.blogspot.com/2018/12/how-to-use-stbimagewrite.html
* [How Blurs & Filters Work](https://www.youtube.com/watch?v=C_zFhWdM4ic)
* https://elcharolin.wordpress.com/2017/03/24/gaussian-blur-with-opencl/
* https://www.eriksmistad.no/gaussian-blur-using-opencl-and-the-built-in-images-textures/
* https://stackoverflow.com/questions/54614167/trying-to-implement-gaussian-filter-in-c
* https://lisyarus.github.io/blog/graphics/2022/04/21/compute-blur.html