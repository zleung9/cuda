#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 10000000
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8


void add_cpu_1d(float *a, float *b, float *c, int n) {
    for (int i; i<n; i++) {
        c[i] = a[i] + b[i];
    }
}


__global__ void add_gpu_1d(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx  + k * nx * ny;
        if (idx < nx * ny * nz) {
            c[i] = a[i] + b[i]
        }
    } 
}



void init_vector(float *vec, int n) {
    for (int i; i<n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main() {

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initiate vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    

    // cpu vector add
    for (int i; i<20; i++ ) {
        add_cpu_1d(h_a, h_b, h_c, N);
    }

    // Defie the grid and the number of blocks needed to GPU calculation
    int num_blocks = (N - 1) / BLOCK_SIZE_1D + 1;
    for (int i; i<20; i++) {
        add_gpu_1d<<<numb_blocks, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N)
    }

    // Define the grid and the number of blocks
    int nx = 100, ny = 100, nz = 1000;
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z)
    dim3 num_blocks_3d(
        (nx - 1) / block_size_3d.x + 1;
        (ny - 1) / block_size_3d.y + 1;
        (nz - 1) / block_size_3d.z + 1;
    )

    for (int i; i<20; i++) {
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c, nx, ny, nz)
    }

}
