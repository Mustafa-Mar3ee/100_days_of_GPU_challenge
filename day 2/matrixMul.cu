#include <iostream>
#include <cuda_runtime.h>

#define N 2 // Matrix size (N x N)

// CUDA kernel for matrix multiplication
__global__ void matrixMul(int *A, int *B, int *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        int sum = 0;
        for (int k = 0; k < n; k++)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main()
{

    int size = N * N * sizeof(int);

    // Host matrices
    int h_A[N * N] = {1, 2, 3, 4};
    int h_B[N * N] = {5, 6, 7, 8};
    int h_C[N * N];

    // Device matrices
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);

    // Launch kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result Matrix C:\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    // Free Device Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free Host Memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
