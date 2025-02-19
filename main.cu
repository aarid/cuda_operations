#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>

// Structure for benchmarking results
struct BenchmarkResult {
    double cpuTime;
    double gpuTime;
    double maxDiff;
};

// CUDA error checking
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// CPU matrix multiplication
void cpuMatrixMultiply(const std::vector<float>& A,
                      const std::vector<float>& B,
                      std::vector<float>& C,
                      int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// GPU matrix multiplication kernel
__global__
void gpuMatrixMultiply(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU 2D convolution
void cpuConv2D(const std::vector<float>& input,
               const std::vector<float>& kernel,
               std::vector<float>& output,
               int inputSize, int kernelSize) {
    int outputSize = inputSize - kernelSize + 1;
    for (int y = 0; y < outputSize; ++y) {
        for (int x = 0; x < outputSize; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    sum += input[(y + ky) * inputSize + (x + kx)] *
                           kernel[ky * kernelSize + kx];
                }
            }
            output[y * outputSize + x] = sum;
        }
    }
}

// GPU 2D convolution kernel
__global__
void gpuConv2D(const float* input, const float* kernel, float* output,
               int inputSize, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outputSize = inputSize - kernelSize + 1;

    if (x < outputSize && y < outputSize) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
                sum += input[(y + ky) * inputSize + (x + kx)] *
                       kernel[ky * kernelSize + kx];
            }
        }
        output[y * outputSize + x] = sum;
    }
}

// Save matrix to CSV for visualization
void saveToCSV(const std::vector<float>& matrix, int rows, int cols, const std::string& filename) {
    std::ofstream file(filename);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i * cols + j];
            if (j < cols - 1) file << ",";
        }
        file << "\n";
    }
}

// Benchmark matrix multiplication
BenchmarkResult benchmarkMatMul(int N) {
    BenchmarkResult result;
    std::vector<float> A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);
    
    // Initialize with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // CPU multiplication
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuMatrixMultiply(A, B, C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    result.cpuTime = std::chrono::duration<double>(end_cpu - start_cpu).count();

    // GPU multiplication
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, N * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    gpuMatrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_gpu = std::chrono::high_resolution_clock::now();
    result.gpuTime = std::chrono::duration<double>(end_gpu - start_gpu).count();

    CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    result.maxDiff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        result.maxDiff = max(result.maxDiff, std::abs(C_cpu[i] - C_gpu[i]));
    }

    // Save sample results for visualization
    if (N <= 32) {  // Save only for small matrices
        saveToCSV(A, N, N, "csv_files/matrix_A.csv");
        saveToCSV(B, N, N, "csv_files/matrix_B.csv");
        saveToCSV(C_gpu, N, N, "csv_files/matrix_C.csv");
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return result;
}

// Benchmark 2D convolution
BenchmarkResult benchmarkConv2D(int inputSize, int kernelSize) {
    BenchmarkResult result;
    int outputSize = inputSize - kernelSize + 1;
    
    std::vector<float> input(inputSize * inputSize);
    std::vector<float> kernel(kernelSize * kernelSize);
    std::vector<float> output_cpu(outputSize * outputSize);
    std::vector<float> output_gpu(outputSize * outputSize);

    // Initialize with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : input) val = dist(rng);
    for (auto& val : kernel) val = dist(rng);

    // CPU convolution
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuConv2D(input, kernel, output_cpu, inputSize, kernelSize);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    result.cpuTime = std::chrono::duration<double>(end_cpu - start_cpu).count();

    // GPU convolution
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_gpu.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((outputSize + blockSize.x - 1) / blockSize.x,
                  (outputSize + blockSize.y - 1) / blockSize.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    gpuConv2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, inputSize, kernelSize);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_gpu = std::chrono::high_resolution_clock::now();
    result.gpuTime = std::chrono::duration<double>(end_gpu - start_gpu).count();

    CHECK_CUDA(cudaMemcpy(output_gpu.data(), d_output, output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    result.maxDiff = 0.0;
    for (size_t i = 0; i < output_gpu.size(); ++i) {
        result.maxDiff = max(result.maxDiff, std::abs(output_cpu[i] - output_gpu[i]));
    }

    // Save sample results for visualization
    if (inputSize <= 32) {
        saveToCSV(input, inputSize, inputSize, "csv_files/conv_input.csv");
        saveToCSV(kernel, kernelSize, kernelSize, "csv_files/conv_kernel.csv");
        saveToCSV(output_gpu, outputSize, outputSize, "csv_files/conv_output.csv");
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output));

    return result;
}

int main() {
    // Test matrix multiplication
    std::cout << "Matrix Multiplication Benchmark\n";
    std::cout << "-------------------------------\n";
    for (int N : {32, 128, 512, 1024}) {
        std::cout << "\nSize: " << N << "x" << N << std::endl;
        auto result = benchmarkMatMul(N);
        std::cout << "CPU Time: " << result.cpuTime << "s\n"
                  << "GPU Time: " << result.gpuTime << "s\n"
                  << "Speedup: " << result.cpuTime/result.gpuTime << "x\n"
                  << "Max Difference: " << result.maxDiff << std::endl;
    }

    // Test 2D convolution
    std::cout << "\n2D Convolution Benchmark\n";
    std::cout << "------------------------\n";
    for (int size : {32, 128, 512, 1024}) {
        std::cout << "\nInput Size: " << size << "x" << size 
                  << ", Kernel: 3x3" << std::endl;
        auto result = benchmarkConv2D(size, 3);
        std::cout << "CPU Time: " << result.cpuTime << "s\n"
                  << "GPU Time: " << result.gpuTime << "s\n"
                  << "Speedup: " << result.cpuTime/result.gpuTime << "x\n"
                  << "Max Difference: " << result.maxDiff << std::endl;
    }

    return 0;
}