#include <stdio.h>
#include <string.h>
#include <time.h>

//define the GPU computed function 
__global__ void saxpy_gpu(int N, float a, float b, float* x){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) 
		x[i] = a*x[i] + b;
	
}

//define the CPU computed function
__host__ void saxpy_cpu(int N, float a, float b, float* x){

	int i; 
	for(i=0;i<N;i++){
			x[i] = a*x[i] + b;
	}
}

int main(int argc, char *argv[]){

	int i;
	int N = atoi(argv[1]);
	float a = atof(argv[2]); 
	float b = atof(argv[3]);
	int numBytes = N*sizeof(float);

	// defining the cpu and the GPU objects
	float* x_cpu = (float *)malloc(numBytes);
	float* x_GPU;
	
	// init the array a_cpu
	for(i=0;i<N;i++){
		x_cpu[i] = i;
	}
	
	// Get current time
	clock_t begin = clock();
	clock_t end;
	// Start the computing
	if(strcmp(argv[4], "GPU")==0){
		// Memory allocation for the array on the GPU
		cudaMalloc((void**)&x_GPU, numBytes);
		cudaMemcpy(x_GPU, x_cpu, numBytes, cudaMemcpyHostToDevice);
	
		const int nThreadsPerBlocks  = (argc==6)? atoi(argv[5]): 512;
    		const int nBlocks = (N / nThreadsPerBlocks) + ( (N % nThreadsPerBlocks) == 0 ? 0 : 1);
		
    		saxpy_gpu<<<nBlocks, nThreadsPerBlocks>>>(N, a, b, x_GPU);		

		cudaMemcpy(x_cpu, x_GPU, numBytes, cudaMemcpyDeviceToHost);
		end = clock();

	}else if(strcmp(argv[4], "CPU")==0){
		saxpy_cpu(N, a, b, x_cpu);
		end = clock();
	}else{
		printf("Please check your compute mode");
		return EXIT_FAILURE;
	}
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Elapsed: %f seconds\n", time_spent);
	for(i = 0; i < 10; i++)
             printf("results %d : %f \n", i, x_cpu[i]);
	for(i = 10; i > 0; i--)
             printf("results %d : %f \n", N-i, x_cpu[N-i]);
 	return 0;

}

