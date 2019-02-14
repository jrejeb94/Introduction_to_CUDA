#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <math.h>
#include <ctype.h>
#define PI 3.141592

void write_file(char* filename, int size, float* image){
	
	// writing the file 
	int i;
	FILE* fptr = fopen(filename, "w");
	if(fptr == NULL){
	        printf("Error: could not open the image file\n");
	}
	fprintf(fptr, "P2\n");
	fprintf(fptr, "%d %d\n", size, size);
	for(i = 0; i < size*size; i++){
		fprintf(fptr, "%d\n", (int)image[i]);
	}
	fclose(fptr);
}

struct Myimage
{
	int size;
	int * image;
};

__host__ Myimage read_file(char* filename){
	
	
	FILE* file = fopen(filename, "r");
	char* n;
	int i;
	Myimage img;
	int j = 0;
	if(file == NULL){
        	printf("Error: could not open the image file\n");
    	}
	fscanf(file, "%s", &n);
  	fscanf (file, "%d %d", &img.size, &img.size); 
	 
	img.image = (int *)malloc(img.size * img.size * sizeof(int));          	  	
	while (!feof (file)){  
      		fscanf(file, "%d", &i);	
		img.image[j] = i;		 
		j++; 
    	}
  	fclose (file);
	return img;
}

__host__  void kernel_gauss(int s, float* w){	
	int y;
	for (y=0; y<s; y++){
		w[y] = 1/(s*sqrt(2*PI))*exp(-y/(2*(s^2)));
	}
}

__host__  void kernel_boite(int s, float* w){	
	int y;
	for (y=0; y<s; y++){
		w[y] = 1;
	}
}

__host__  void kernel_bilateral(int r, float* w){
	
	int y;
	for (y=0; y<256; y++){
		w[y] = 1/(r*sqrt(2*PI))*exp(-(y^2)/(2*(r^2)));
	}
}

__global__ void Filtre(int N, int s, float* S, float* w){	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i<N){
		float sum1 = 0;
		float sum2 = 0;
		int y1;
		int y2;
		int x1 = i/(N+1);
		int x2 = i%N;
		for (y1=x1-s; y1<x1+s; y1++){
			for (y2=x2-s; y2<x2+s; y2++){
				sum1 += S[abs(y1*N+y2)]*w[(x1-y1)^2 + (x2-y2)^2]; 
				sum2 += w[(x1-y1)^2 + (x2-y2)^2];
			}
		}
		S[i]= sum1/sum2;
	}	 
}

__global__ void Filtre_bilateral(int N, int s, float* S, float* w, float* w_r){	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i<N){
		float sum1 = 0;
		float sum2 = 0;
		int y1;
		int y2;
		int x1 = i/(N+1);
		int x2 = i%N;
		for (y1=x1-s; y1<x1+s; y1++){
			for (y2=x2-s; y2<x2+s; y2++){
				// check if in the support of Wr ???!!
				sum1 += S[abs(y1*N+y2)]*w[(x1-y1)^2 + (x2-y2)^2]*w_r[(int)abs(S[i]-S[y1*N+y2])]; 
				sum2 += w[(x1-y1)^2 + (x2-y2)^2]*w_r[(int)abs(S[i]-S[y1*N+y2])];
			}
		}
		S[i]= sum1/sum2;
	}	 
}


int main(int argc, char *argv[]){
	
	int s = atoi(argv[1]);
	int r = 50;
	char* kernel = argv[2];
	char* input_image = argv[3];
	char* output_image = argv[4];

	Myimage img = read_file(input_image);
	
	int numFilter = s*sizeof(float);

	int N=img.size;
	int numBytes = N*sizeof(float);
	float* S_cpu=(float *)malloc(N * N * sizeof(float));
	for(int i = 0; i < N*N; ++i) {
    		S_cpu[i] = (float)img.image[i];
	}
	float* S_GPU;

	cudaMalloc((void**)&S_GPU, numBytes);
	cudaMemcpy(S_GPU, S_cpu, numBytes, cudaMemcpyHostToDevice);

	float* w_cpu= (float*) calloc(s, sizeof(float));
	float* w_GPU;

	if (strcmp(kernel, "g")==0){
		kernel_gauss(s, w_cpu);	
	}else if(strcmp(kernel, "b")==0){
			kernel_boite(s, w_cpu);
	}else if(strcmp(kernel, "bl")==0){
		int numFilter_bl = 256*sizeof(float);
		//float* w_r_cpu= (float*) calloc(256, sizeof(float));
		float w_r_cpu[256];
		float w_r_GPU[256];
		
		kernel_boite(s, w_cpu);
		kernel_bilateral(r, w_r_cpu);

		cudaMalloc((void**)&w_GPU, numFilter);
		cudaMemcpy(w_GPU, w_cpu, numFilter, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&w_r_GPU, numFilter_bl);
		cudaMemcpy(w_r_GPU, w_r_cpu, numFilter_bl, cudaMemcpyHostToDevice);
		int nblocks = (N + 255)/256;
		Filtre_bilateral<<<nblocks,256>>>(N, s, S_GPU, w_GPU, w_r_GPU);
		cudaMemcpy(S_cpu, S_GPU, numBytes, cudaMemcpyDeviceToHost);

		write_file(output_image, N, S_cpu);
		
		return 0;
	}else{
		printf("Please recheck the kernel type! Value must be equal to : b for 'boite' or g for 'gaussian'.");
		return EXIT_FAILURE;
	}
	

	cudaMalloc((void**)&w_GPU, numFilter);
	cudaMemcpy(w_GPU, w_cpu, numFilter, cudaMemcpyHostToDevice);

	int nblocks = (N + 255)/256;
	Filtre<<<nblocks,256>>>(N, s, S_GPU, w_GPU);
	cudaMemcpy(S_cpu, S_GPU, numBytes, cudaMemcpyDeviceToHost);
	
	write_file(output_image, N, S_cpu);
	return 0;
}
