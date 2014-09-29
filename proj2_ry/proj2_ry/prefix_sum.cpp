#include <stdio.h>
#include <iostream>

//void scan_cpu( float*& output, float* input, int length) 
//{ 
//	 output[0] = 0;
//	 for(int j = 1; j < length; ++j) 
//	 { 
//		output[j] = input[j-1] + output[j-1]; 
//	 } 
//
//	std::cout<<"Serial Scan Output:"<<std::endl;
//	for(int i = 0; i<length; i++)
//	{
//		std::cout<<output[i]<<" ";
//	}
//	std::cout<<std::endl;
//} 
//
//void scatter_cpu(float*& output, float* input, int length)
//{
//	int count = 0;
//	for(int i = 0; i < length; i++)
//	{
//		if(int(input[i])%2 == 1 ) //odd
//		{
//			output[i] = 1;
//			count++;
//		}else{
//			output[i] = 0;
//		}
//	}
//	std::cout<<"Serial Scatter Output:"<<std::endl;
//	for(int i = 0; i<count; i++)
//	{
//		std::cout<<output[i]<<" ";
//	}
//	std::cout<<std::endl;
//}

//void main()
//{
//	int arraySize = 10;
//	std::cout<<"Array Size:"<<std::endl;
//	std::cin>>arraySize;
//	float* input = new float[arraySize];
//	std::cout<<"Input:"<<std::endl;
//	for(int i = 0; i<arraySize; i++)
//	{
//		std::cin>>input[i];
//	}
//	float* output = new float[arraySize];
//	scan_cpu(output, input, arraySize);
//	std::cout<<"Output:"<<std::endl;
//	for(int i = 0; i<arraySize; i++)
//	{
//		std::cout<<output[i]<<" ";
//	}
//

//}