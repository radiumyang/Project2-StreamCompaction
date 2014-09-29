#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/copy.h>

#define BLOCKSIZE 128
#define ARRAYSIZE 600

__global__ void NaiveKernel(float* inputd, float* outputd, float* temp, int d, int n, int swap)
{
	int tx =  blockDim.x * blockIdx.x + threadIdx.x;

	int thr = pow(2.0f,d-1);

	if(tx == 0) outputd[tx] = temp[tx];
	if(tx >= thr && tx < n)
		//for (int k = thr; k < n; ++k)
		{
		/*	if(swap == 0){
			outputd[tx] = inputd[tx - thr] + inputd[tx];
			}else if (swap == 1){*/
			outputd[tx] = temp[tx - thr] + temp[tx];
		//		inputd[tx] = outputd[tx-thr] + outputd[tx];
		//	}
		}

}

void naive_prefix_sum (float *& output, float *input, int n)
{
	int loopNum = floor(log(float(n))/log(2.0f));

	float* outputd, *inputd, *tempd;
	
	int size = n*sizeof(float);
	cudaMalloc((void**)&inputd, size);
	cudaMemcpy(inputd, input, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(ceil((float)n/BLOCKSIZE));
	cudaMalloc((void**)&outputd, size);

	cudaMemcpy(outputd, output, size, cudaMemcpyDeviceToHost);

	int swap = 0;
	float* temp = new float[n];
	temp[0] = 0;
	for(int l = 1; l<=n; l++) 
	{
		temp[l] = input[l-1];
	//	std::cout<<"tmp"<<temp[l]<<std::endl;
	}
	cudaMalloc((void**)&tempd, size);
	cudaMemcpy(tempd, temp, size, cudaMemcpyHostToDevice);
	
	// check runtime...
	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);
	for(int d = 1; d<=loopNum; d++){
		
		NaiveKernel<<<dimGrid, dimBlock>>>(inputd, outputd, tempd, d, n, swap);
		cudaThreadSynchronize();
		cudaMemcpy(output, outputd, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(temp, tempd, size, cudaMemcpyDeviceToHost);
		
	//	std::cout<<"Iteration "<<d<<":"<<std::endl;
		for(int l = 0; l<n; l++) 
		{
	//			std::cout<<output[l]<<" ";
				temp[l] = output[l];
		}


		cudaMemcpy(tempd, temp, size, cudaMemcpyHostToDevice);
		cudaMemcpy(outputd, output, size, cudaMemcpyHostToDevice);
	//	std::cout<<std::endl;
		swap = 1;
	}
	
	cudaMemcpy(output, outputd, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!

	std::cout<<std::endl;
	std::cout<<"Naive Prefix Sum Runtime: "<<elapsedTime<< ";  ArraySize: "<<ARRAYSIZE<<std::endl;
	// Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(outputd); cudaFree(inputd);

	std::cout<<"Naive Prefix Sum Output:"<<std::endl;
//	int length = n;
//	if(BLOCKSIZE < n) length = BLOCKSIZE;
	for(int i = 0; i<n; i++)
	{
		std::cout<<output[i]<<" ";
	}
	std::cout<<std::endl;
}


__global__ void SharedKernel(float * input, float *output, int n) 
{ 
	__shared__ float* tmp1, *tmp2; 
	tmp1 = new float[n+1];
	tmp2 = new float[n+1];
	int tx = threadIdx.x; 

	// This is exclusive scan, so shift right by one and set first elt to 0 
	 tmp1[tx] = (tx > 0) ? input[tx-1] : 0; 
	 tmp2[tx] = (tx > 0) ? input[tx-1] : 0; 

	__syncthreads(); 
	if(tx < n)
	{
		 for (int offset = 1; offset < n; offset *= 2) 
		 { 
			 if (tx >= offset) 
			 {
				tmp2[tx] = tmp1[tx-offset]+tmp1[tx]; 
			 }
			 else {
				tmp2[tx] = tmp1[tx]; 
			 }
			 tmp1[tx] = tmp2[tx];
			 __syncthreads(); 
		 } 
	}
	output[tx] = tmp2[tx]; // write output 
} 

void shared_prefix_sum (float * output, float *input, int n)
{
	int loopNum = floor(log(float(n))/log(2.0f));

	float* outputd, *inputd;
	
	int size = n*sizeof(float);
	cudaMalloc((void**)&inputd, size);
	cudaMemcpy(inputd, input, size, cudaMemcpyHostToDevice);
		
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(1);

	cudaMalloc((void**)&outputd, size);
	
	// check runtime...
	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);

	SharedKernel<<<dimGrid, dimBlock, BLOCKSIZE*sizeof(float)>>>(inputd, outputd, n);

	cudaMemcpy(output, outputd, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!

	std::cout<<std::endl;
	std::cout<<"Shared Memory Single-Block Prefix Sum Runtime: "<<elapsedTime<< ";  ArraySize: "<<ARRAYSIZE<<std::endl;
	// Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(outputd); cudaFree(inputd);

	std::cout<<"Shared Memory Prefix Sum Output:"<<std::endl;
	int length = n;
	if(BLOCKSIZE < n) 
	{
		length = BLOCKSIZE;
		std::cout<<"!!!Arraysize is over Blocksize!!!"<<std::endl;
	}
	for(int i = 0; i<length; i++)
	{
		std::cout<<output[i]<<" ";
	}
	std::cout<<std::endl;
}

__global__ void SharedGeneralKernel(float * input, float *output, float *sum, int n) 
{ 
	__shared__ float* tmp1, *tmp2; 
	tmp1 = new float[n+1];
	tmp2 = new float[n+1];
	int index = blockDim.x*blockIdx.x+threadIdx.x; 
	int tx = threadIdx.x;

	// This is exclusive scan, so shift right by one and set first elt to 0 
	if(index < n)
	{ 
		tmp1[tx] = (index > 0) ? input[index-1] : 0; 
		tmp2[tx] = (index > 0) ? input[index-1] : 0; 

		__syncthreads(); 
	
		 for (int offset = 1; offset < n; offset *= 2) 
		 { 
			 if (tx >= offset) 
			 {
				tmp2[tx] = tmp1[tx-offset]+tmp1[tx]; 
			 }
			 else {
				tmp2[tx] = tmp1[tx]; 
			 }
			 tmp1[tx] = tmp2[tx];
			 __syncthreads(); 
		 } 
	}	
	output[index] = tmp2[tx]; // write output 
	sum[blockIdx.x] = tmp2[blockDim.x-1]; //last number is the sum of current block
} 

__global__ void SharedGeneralKernel_plus(float * input, float *output, float *sum, int n) 
{
	int index = blockDim.x*blockIdx.x+threadIdx.x; 
	if(blockIdx.x > 0)
	{
		output[index] += sum[blockIdx.x - 1];
	}
}

void shared_prefix_sum_general(float*& output, float* input, int n)
{
	int loopNum = floor(log(float(n))/log(2.0f));
	int blocknum = ceil(float(n)/BLOCKSIZE);
	float* sum = new float[blocknum];

	float* outputd, *inputd, *sumd;
	
	int size = n*sizeof(float);
	cudaMalloc((void**)&inputd, size);
	cudaMemcpy(inputd, input, size, cudaMemcpyHostToDevice);
		
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(blocknum);

	int size2 = blocknum * sizeof(float);
	cudaMalloc((void**)&outputd, size);
	cudaMalloc((void**)&sumd, size2);

	// check runtime...
	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);


	SharedGeneralKernel<<<dimGrid, dimBlock, n*sizeof(float)>>>(inputd, outputd, sumd, n);
	SharedGeneralKernel_plus<<<dimGrid, dimBlock>>>(inputd, outputd, sumd, n);
	cudaMemcpy(sum, sumd, size2, cudaMemcpyDeviceToHost);
	cudaMemcpy(output, outputd, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); 

	std::cout<<std::endl;
	std::cout<<"Shared Memory General Prefix Sum Runtime: "<<elapsedTime<< ";  ArraySize: "<<ARRAYSIZE<<std::endl;
	// Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(outputd); cudaFree(inputd); cudaFree(sumd);

	//std::cout<<"Sum:"<<std::endl;
	//for(int i = 0; i<blocknum; i++)
	//{
	//	std::cout<<sum[i]<<" ";
	//}
	std::cout<<std::endl;

	std::cout<<"Shared Memory GENERAL Prefix Sum Output:"<<std::endl;
	for(int i = 0; i<n; i++)
	{
		std::cout<<output[i]<<" ";
	}
	std::cout<<std::endl;
}

__global__ void ScatterKernel(float * inputd, float* outputd, int n)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < n)
	{		
		if(int(inputd[index])%2 == 1){
			// if it is odd
			outputd[index] = 1;
		}
		else{
			outputd[index] = 0;
		}
	}
}

void stream_compact(float* input, float*& output, float* sctarray, int n)
{
	float*inputd, *outputd;
	float* sctarrayd;
	int size = n*sizeof(float);
	
	cudaMalloc((void**)&inputd, size);
	cudaMemcpy(inputd, input, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&sctarrayd, size);
//	cudaMalloc((void**)&outputd, size);
	//first, Scatter
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(ceil((float)n/BLOCKSIZE));
	ScatterKernel<<<dimGrid, dimBlock>>>(inputd, sctarrayd, n);
	cudaMemcpy(sctarray, sctarrayd, size, cudaMemcpyDeviceToHost);
	std::cout<<"Scatter Array Output:"<<std::endl;
	for(int i = 0; i<n; i++)
	{
		std::cout<<sctarray[i]<<" ";
	}	
	std::cout<<std::endl;
	//then, scan
	std::cout<<"Stream Compact Using Naive Prefix Sum:"<<std::endl;
	naive_prefix_sum(output, sctarray, n);

	//cudaMemcpy(output, outputd, size, cudaMemcpyDeviceToHost);
	//cudaFree(outputd); 
	cudaFree(inputd); cudaFree(sctarrayd);
}

struct is_odd : public thrust::unary_function<float, bool>
{
    __host__ __device__
    bool operator()(float x)
    {
        return int(x)%2;
    }
};

void stream_compact_thrust(float* input, float*& output, float* sctarray, int n)
{
	float *p = &input[0];
	int length = thrust::count_if(p, p+n, is_odd());
	output = new float[length];
	thrust::copy_if(p, p+n, output, is_odd());

//	new_length = length;

	std::cout<<"Output:"<<std::endl;
	for(int i = 0; i<length; i++)
	{
		std::cout<<output[i]<<" ";
	}	
}

void scan_cpu( float*& output, float* input, int length) 
{ 
	 output[0] = 0;
	 for(int j = 1; j < length; ++j) 
	 { 
		output[j] = input[j-1] + output[j-1]; 
	 } 

	std::cout<<"Serial Scan Output:"<<std::endl;
	for(int i = 0; i<length; i++)
	{
		std::cout<<output[i]<<" ";
	}
	std::cout<<std::endl;
} 

void scatter_cpu(float*& output, float* input, int length)
{
	int count = 0;
	for(int i = 0; i < length; i++)
	{
		if(int(input[i])%2 == 1 ) //odd
		{
			output[i] = 1;
			count++;
		}else{
			output[i] = 0;
		}
	}
	std::cout<<"Serial Scatter Output:"<<std::endl;
	for(int i = 0; i<count; i++)
	{
		std::cout<<output[i]<<" ";
	}
	std::cout<<std::endl;
}

void init(float* input, float* output, int arraySize)
{
	for(int i = 0; i<arraySize; i++)
	{
		input[i] = i;
	}
	delete[] output;
	output = new float[arraySize];
}

void init(float* input, float* output, float* sctarray, int arraySize)
{
	for(int i = 0; i<arraySize; i++)
	{
		input[i] = i;
	}
	delete[] output;
	output = new float[arraySize];
	delete[] sctarray;
	sctarray = new float[arraySize];
}

void main()
{
	int arraySize = ARRAYSIZE;
	/*std::cout<<"Array Size:"<<std::endl;
	std::cin>>arraySize;*/
	float* input = new float[arraySize];

	std::cout<<"Input:"<<std::endl;
	for(int i = 0; i<arraySize; i++)
	{
		input[i] = i;
		std::cout << input[i] << " ";
	}
	std::cout << std::endl;

	float* output = new float[arraySize];
	float* sctarray = new float[arraySize];

	std::cout<<"*****************************************"<<std::endl;
	std::cout<<"******************CPU******************"<<std::endl;

	std::cout << "************************************" << std::endl;
	std::cout << "************ PART 1 **************" << std::endl;
	scan_cpu(output, input, arraySize);
	init(input, output, arraySize);
	
	std::cout << "************************************" << std::endl;
	std::cout << "************ PART 4.1 **************" << std::endl;
	scatter_cpu(output, input, arraySize);
	init(input, output, arraySize);
	

	std::cout<<"*****************************************"<<std::endl;
	std::cout<<"******************GPU******************"<<std::endl;

	std::cout << "************************************" << std::endl;
	std::cout << "************ PART 2 **************" << std::endl;
    std::cout << "***Naive Prefix Sum***" << std::endl;
	naive_prefix_sum(output, input, arraySize);
	init(input, output, arraySize);

	std::cout << "*************************************" << std::endl;
	std::cout << "************ PART 3A **************" << std::endl;
    std::cout << "***Shared Memory Prefix Sum, Single Block***" << std::endl;
	shared_prefix_sum(output, input, arraySize);
	init(input, output, arraySize);

	std::cout << "*************************************" << std::endl;
	std::cout << "************ PART 3B **************" << std::endl;
    std::cout << "***Shared Memory Prefix Sum, General***" << std::endl;
	shared_prefix_sum_general(output, input, arraySize);
	init(input, output, arraySize);

	std::cout << "************************************" << std::endl;
	std::cout << "************ PART 4.2 **************" << std::endl;
    std::cout << "***GPU Stream Compact***" << std::endl;
	stream_compact(input, output, sctarray, arraySize);
	init(input, output, sctarray, arraySize);

	std::cout << "*****************************************" << std::endl;
	std::cout << "*************** PART 4.3 **************" << std::endl;
    std::cout << "***GPU Stream Compact: Thrust***" << std::endl;

	stream_compact_thrust(input, output, sctarray, arraySize);


	int test;
}