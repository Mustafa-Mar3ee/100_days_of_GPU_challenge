#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(int *a, int *b, int *c, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
    {
        c[index] = a[index] + b[index];
    }
}

int main()
{
    int N = 1000;
    int size = N * sizeof(int);
    int *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);

    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory on GPU
    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy Data from HOST (CPU) to Device (GPU)

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blockPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blockPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Vector Addition Rsults (First 10 Elements):\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFree(h_b);
    cudaFree(h_c);

    return 0;
}