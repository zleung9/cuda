#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 400000000  // Vector size = 10 million
#define BLOCK_SIZE 256 // Number of threads per block

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i=0; i<n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i=0; i<n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed (BLOCK_SIZE: threads per block)
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // If, say, N=1025, then num_blocks = 5 (1 extra bit needs an entire block)
    // (1024 + 256 - 1) / 256 = 1279 / 256 = 4.999 -> truncated to 4
    // (1025 + 256 - 1) / 256 = 1280 / 256 = 5 -> no truncation
    // (1026 + 256 - 1) / 256 = 1281 / 256 = 5.0003 -> truncated to 5
    

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i=0; i<3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    double start_time, end_time, total_time;
    double cpu_avg_time, gpu_avg_time;
    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    total_time = 0.0;
    for (int i=0; i<20; i++) {
        start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        end_time = get_time();
        total_time += end_time - start_time;
    }
    cpu_avg_time = total_time / 20;
    printf("Average CPU time: %f ms\n", cpu_avg_time * 1e3);

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    total_time = 0.0;
    for (int i=0; i<20; i++) {
        start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        end_time = get_time();
        total_time += end_time - start_time;
    }
    gpu_avg_time = total_time / 20;
    printf("Average GPU time: %f ms\n", gpu_avg_time * 1e3);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Verify results
    start_time = get_time();
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    end_time = get_time();
    printf("Time to copy data from device to host: %f ms\n", (end_time - start_time) * 1e3);
    bool correct = true;
    for (int i=0; i<N; i++) {
        if (fabs(h_c_gpu[i] - h_c_cpu[i]) > 1e5) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
