#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INTERVALLE 1
#define PI 3.141592

__host__  void kernel_gauss(int s, float* w){	
	int y;
	for (y=0; y<s; y++){
		w[y] = 1/(s*sqrt(2*PI))*exp(-(y^2)/(2*s^2));
	}	 
}

__host__  void kernel_boite(int s, float* w){	
	int y;
	for (y=0; y<s; y++){
		w[y] = 1;
	}	 
}

__global__ void Filtre_gpu(int N, int s, float* x, float* w){	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i<N){
		float sum1 = 0;
		float sum2 = 0;
		int y;
		for (y=i-s; y<i+s; y++){
			
			sum1 += x[abs(y)]*w[abs(i-y)]; 
			sum2 += w[abs(i-y)];
			
		}
		x[i]= sum1/sum2;
	}
		 
}

__host__ void Filtre_cpu(int N, int s, float* x, float* w){

	float sum1 = 0;
	float sum2 = 0;
	int y;
	int i; 
	for(i=0;i<N;i++){
		for (y=i-s; y<i+s; y++){
			sum1 += x[min(abs(y), 2*N-y-2)]*w[abs(i-y)]; 
			sum2 += w[abs(i-y)];
		}
		x[i]= sum1/sum2;
	}
}

int main(int argc, char *argv[]){

	int N = atoi(argv[1]);
	int s = atof(argv[2]); 
	char* kernel = argv[3];
	char* output_file = argv[5];

	int numBytes = N*sizeof(float);
	int numFilter = s*sizeof(float);

	float* S_cpu = (float *)malloc(numBytes);
	float* S_GPU;

	float* w_cpu= (float*) calloc(s, sizeof(float));
	float* w_GPU;

	double f1 = 0.005;
	double f2 = 0.05;
	double A1 = 50;
	double A2 = 5;
	double t = 1;
	int i;

	// Constructing the signal
	for(i = 0; i < N; i++){
		S_cpu[i] = A1 * sin(2*PI*f1*t) + A2 * sin(2*PI*f2*t);
		t += INTERVALLE;
	}
	
	float* old=(float *)malloc(numBytes);
	for(int i = 0; i < N; i++) {
	        old[i] = S_cpu[i];	
	}
	if (strcmp(kernel, "g")==0){
		kernel_gauss(s, w_cpu);	
	}else if(strcmp(kernel, "b")==0){
			kernel_boite(s, w_cpu);
	}else{
		printf("Please recheck the kernel type! Value must be equal to : b for 'boite' or g for 'gaussian'.");
		return EXIT_FAILURE;
	}

	// Get current time
	clock_t begin = clock();
	clock_t end;

	if(strcmp(argv[4], "GPU")==0){

		cudaMalloc((void**)&S_GPU, numBytes);
		cudaMemcpy(S_GPU, S_cpu, numBytes, cudaMemcpyHostToDevice);
			
		cudaMalloc((void**)&w_GPU, numFilter);
		cudaMemcpy(w_GPU, w_cpu, numFilter, cudaMemcpyHostToDevice);

		int nblocks = (N + 255)/256;
		Filtre_gpu<<<nblocks,256>>>(N, s, S_GPU, w_GPU);
		cudaMemcpy(S_cpu, S_GPU, numBytes, cudaMemcpyDeviceToHost);	
	}else if(strcmp(argv[4], "CPU")==0){
		Filtre_cpu(N, s, S_cpu, w_cpu);
		end = clock();
	}else{
		printf("Please check your compute mode");
		return EXIT_FAILURE;
	}
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Elapsed: %f seconds\n", time_spent);

	// writing the file 
	FILE* fptr = fopen(output_file, "w");
	if(fptr == NULL)
		{
	        	printf("Error: could not open 'signal.data'\n");
	        	return EXIT_FAILURE;
	    	}

	fprintf(fptr, "old_Signal \t filtered_Signal\n");
	for(int i = 0; i < 200; i++){
		fprintf(fptr, "%f \t %f\n", old[i], S_cpu[i]);
	}
	printf("Created %s ....", output_file); 
	fclose(fptr);

 	return 0;

}


