/**************************************************************************
*   文件名：Sobel_4b.cu
*   作者：  孙霖(SC19023100)(Seafood)
*   说明：  cudau作业之Sobel算子边缘检测第四版
*   将数组存入纹理内存
*   
****************************************************************************/
//----------------------------头文件包含和空间声明------------------------------------
#include "cuda_runtime.h"
#include "cuda.h"
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

typedef unsigned char uchar;

//命名空间
using namespace std;
using namespace cv;

//#define DIM 2048
//----------------------------全局变量---------------------------------------------
Mat g_sobelGradient_X, g_sobelGradient_Y;   //opencv包使用的全局变量
Mat g_sobelAbsGradient_X, g_sobelAbsGradient_Y; //opencv包使用的全局变量
int g_sobelKernelSize = 1;


Mat dst_Cpu, g_gaussImage, g_grayImage, g_dstImage; //图像读取+高斯滤波

int g_imgHeight, g_imgWidth;    //图像的大小Size


int sobel_x[3][3];
int sobel_y[3][3];
int number = 1025;
//cuda的常量内存
__constant__ int dev_sobel_x[3][3];
__constant__ int dev_sobel_y[3][3];

//纹理内存
texture<unsigned char, 1> texIn;
/**
*@author：Seafood
*@name：sobelInCuda()
*@return:void
*@function：使用Cuda对图像进行Sobel边缘检测
*@para：*dataIn:输入图像 *dataOut:输出图像 imgHeight：图像的高 imgWidth:图像的宽
*其他要注意的地方
**/
//Sobel算子边缘检测核函数
__global__ void sobelInCuda(unsigned char *dataOut, int imgHeight, int imgWidth)
{
    //用单thread操作
    //int index = threadIdx.x + blockIdx.x * blockDim.x;
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = xIndex + yIndex * imgWidth;
    //printf("blockDim: %d, gridDim: %d\n", blockDim.x, gridDim.x);
    //printf("xIndex : %d,yIndex : %d,Index : %d\n",xIndex, yIndex, offset);
    /*
    if(threadIdx.x == 1 && blockIdx.x == 1)
    {
        for(int i = 0;i < 3; i++)
        {
            printf("%d %d %d\n", dev_sobel_y[i][0],dev_sobel_y[i][1],dev_sobel_y[i][2]);
        }
        printf("gpu用的高斯图像%s\n", tex1Dfetch(texIn, 1025*imgWidth + 1025));
    }*/
    
    int tex_00,tex_01,tex_02,tex_10,tex_11,tex_12,tex_20,tex_21,tex_22;
    int Gx = 0;
    int Gy = 0;

    //printf("NonTex： %d \nTex: %d \n", dataIn[offset+ imgWidth* 2], tex1D(texIn, offset+ imgWidth* 2));

    while(offset < (imgHeight - 2) * (imgWidth - 2))
    {
        //纹理内存上下左右
        int down_00 = offset;                   int down_01 = down_00 + 1;  int down_02 = down_00 + 2;
        int down_10 = offset + imgWidth;        int down_11 = down_10 + 1;  int down_12 = down_10 + 2;
        int down_20 = offset + imgWidth * 2;    int down_21 = down_20 + 1;  int down_22 = down_20 + 2;

        tex_00 = tex1Dfetch(texIn, down_00);    tex_01 = tex1Dfetch(texIn, down_01);    tex_02 = tex1Dfetch(texIn, down_02);
        tex_10 = tex1Dfetch(texIn, down_10);    tex_11 = tex1Dfetch(texIn, down_11);    tex_12 = tex1Dfetch(texIn, down_12);
        tex_20 = tex1Dfetch(texIn, down_20);    tex_21 = tex1Dfetch(texIn, down_21);    tex_22 = tex1Dfetch(texIn, down_22);
        
        Gx = dev_sobel_x[0][0] * tex_00 + dev_sobel_x[0][1] * tex_01 + dev_sobel_x[0][2] * tex_02
            + dev_sobel_x[1][0] * tex_10 + dev_sobel_x[1][1] * tex_11 + dev_sobel_x[1][2] * tex_12
            + dev_sobel_x[2][0] * tex_20 + dev_sobel_x[2][1] * tex_21 + dev_sobel_x[2][2] * tex_22;
        Gy = dev_sobel_y[0][0] * tex_00 + dev_sobel_y[0][1] * tex_01 + dev_sobel_y[0][2] * tex_02
            + dev_sobel_y[1][0] * tex_10 + dev_sobel_y[1][1] * tex_11 + dev_sobel_y[1][2] * tex_12
            + dev_sobel_y[2][0] * tex_20 + dev_sobel_y[2][1] * tex_21 + dev_sobel_y[2][2] * tex_22;

        int sum = abs(Gx) + abs(Gy);
        //int sum = Gx;
        if (sum > 255)
        {
            sum = 255; //for best performance
        }
        if (sum < 0) sum = 0;
        dataOut[offset] = (int)sum;
        xIndex += blockDim.x * gridDim.x;
        if( xIndex > imgWidth)
        {
            yIndex += blockDim.y * gridDim.y;
            xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        }
        
        //offset = xIndex + yIndex * blockDim.x * gridDim.x;
        offset = xIndex + yIndex * imgWidth;
        //index = xIndex + yIndex * imgWidth;
    }
}

/**
*@author：Seafood
*@name：sobelCpuPixel()
*@return:void016 x 401
*@function：使用CPU对图像像素进行Sobel边缘检测
*@para：None
*其他要注意的地方
**/
//Sobel算子边缘检测CPU函数
void sobelCpuPixel()
{
    clock_t begin_time, end_clock;
    begin_time = clock();

    //CPU用
    Mat img = g_gaussImage;
    Mat newimg = img;
    for (int j = 0; j<img.rows-2; j++)
    {
     for (int i = 0; i<img.cols-2; i++)
     {
         int pixval_x =
         (sobel_x[0][0] * (int)img.at<uchar>(j,i)) + (sobel_x[0][1] * (int)img.at<uchar>(j+1,i)) + (sobel_x[0][2] * (int)img.at<uchar>(j+2,i)) +
         (sobel_x[1][0] * (int)img.at<uchar>(j,i+1)) + (sobel_x[1][1] * (int)img.at<uchar>(j+1,i+1)) + (sobel_x[1][2] * (int)img.at<uchar>(j+2,i+1)) +
         (sobel_x[2][0] * (int)img.at<uchar>(j,i+2)) + (sobel_x[2][1] * (int)img.at<uchar>(j+1,i+2)) + (sobel_x[2][2] * (int)img.at<uchar>(j+2,i+2));
         
         int pixval_y =
         (sobel_y[0][0] * (int)img.at<uchar>(j,i)) + (sobel_y[0][1] * (int)img.at<uchar>(j+1,i)) + (sobel_y[0][2] * (int)img.at<uchar>(j+2,i)) +
         (sobel_y[1][0] * (int)img.at<uchar>(j,i+1)) + (sobel_y[1][1] * (int)img.at<uchar>(j+1,i+1)) + (sobel_y[1][2] * (int)img.at<uchar>(j+2,i+1)) +
         (sobel_y[2][0] * (int)img.at<uchar>(j,i+2)) + (sobel_y[2][1] * (int)img.at<uchar>(j+1,i+2)) + (sobel_y[2][2] * (int)img.at<uchar>(j+2,i+2));
         
         int sum = abs(pixval_x) + abs(pixval_y);
         //int sum = pixval_x;
         if (sum > 255)
         {
             sum = 255; //for best performance
         }
         if (sum < 0) sum = 0;
         newimg.at<uchar>(j,i) = sum;
     }
    }
    end_clock = clock();
    cout << "CPU对像素操作运行时间为: " << static_cast<double>(end_clock - begin_time) / CLOCKS_PER_SEC*1000 << "ms" << endl;//输出运行时间为毫秒
    cvNamedWindow("processed by CPU in Paxel", 0);
    resizeWindow("processed by CPU in Paxel", 800, 600);
    imshow("processed by CPU in Paxel", newimg);
}

/**
*@author：Seafood
*@name：sobelCpuPackage
*@return:void
*@function：使用CPU的OPENCV包对图像进行Sobel边缘检测
*@para：None
*其他要注意的地方
**/
//Sobel算子边缘检测CPU函数

void sobelCpuPackage(int, void*)
{
    clock_t begin_time, end_clock;
    begin_time = clock();
    //求x方向梯度
    Sobel(g_grayImage, g_sobelGradient_X, CV_16S, 1, 0, (2*g_sobelKernelSize + 1), 1, 1, BORDER_DEFAULT);
    convertScaleAbs( g_sobelGradient_X, g_sobelAbsGradient_X);

    //求Y方向梯度
    Sobel(g_grayImage, g_sobelGradient_Y, CV_16S, 1, 0, (2*g_sobelKernelSize + 1), 1, 1, BORDER_DEFAULT);
    convertScaleAbs( g_sobelGradient_Y, g_sobelAbsGradient_Y);

    //合并梯度
    addWeighted(g_sobelAbsGradient_X, 0.5, g_sobelAbsGradient_Y, 0.5,0,dst_Cpu );
    end_clock = clock();
    cout << "CPU对使用OPENCV包操作运行时间为: " << static_cast<double>(end_clock - begin_time) / CLOCKS_PER_SEC*1000 << "ms" << endl;//输出运行时间为毫秒
    cvNamedWindow("CPU处理效果图(OPENCV包)", 0);
    resizeWindow("CPU处理效果图(OPENCV包)", 800, 600);
    imshow("CPU处理效果图(OPENCV包)", dst_Cpu);
}

/**
*@author：Seafood
*@name：pictureInit(void)
*@return:void
*@function：图像读取、高斯滤波和初始化
*@para：None
*其他要注意的地方
**/

//图像读入
void pictureInit(void)
{
    //读入图像
    g_grayImage = imread("rmpicture.png", 0);

    //显示原图
    cvNamedWindow("originimage", 0);
    resizeWindow("originimage", 800, 600);
    imshow("originimage", g_grayImage);
    g_imgHeight = g_grayImage.rows;
    g_imgWidth = g_grayImage.cols;

    // asigning values to sobel x direction
    sobel_x[0][0] = -1; sobel_x[0][1] = 0; sobel_x[0][2] =1;
    sobel_x[1][0] = -2; sobel_x[1][1] = 0; sobel_x[1][2] =2;
    sobel_x[2][0] = -1; sobel_x[2][1] = 0; sobel_x[2][2] =1;
    // asigning values to sobel y direction
    sobel_y[0][0] = -1; sobel_y[0][1] = -2; sobel_y[0][2] = -1;
    sobel_y[1][0] = 0; sobel_y[1][1] = 0; sobel_y[1][2] = 0;
    sobel_y[2][0] = 1; sobel_y[2][1] = 2; sobel_y[2][2] = 1;
    printf(" picture size is %d x %d \n", g_imgHeight, g_imgWidth);
    //高斯滤波
    GaussianBlur(g_grayImage, g_gaussImage, Size(3,3), 0, 0, BORDER_DEFAULT);
    cvNamedWindow("gaussimage", 0);
    resizeWindow("gaussimage", 800, 600);
    imshow("gaussimage", g_gaussImage);
}
/**
*@author：Seafood
*@name：sobelGPUPixel()
*@return:void
*@function：调用GPU对图像处理并计算时间
*@para：None
*其他要注意的地方
**/
//GPU处理函数
void sobelGPUPixel()
{
    //定义变量
    unsigned char *dev_in;
    unsigned char *dev_out;

    Mat dst_Gpu(g_imgHeight, g_imgWidth, CV_8UC1, Scalar(0));
    //创建时间用于计算
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    //开始时间点记录
    cudaEventRecord( start, 0 );
    


    //定义数组
    //申请内存
    cudaMalloc((void**)&dev_in, g_imgHeight * g_imgWidth * sizeof(unsigned char));
    cudaMalloc((void**)&dev_out, g_imgHeight * g_imgWidth * sizeof(unsigned char));
    //纹理内存绑定
    cudaBindTexture(NULL, texIn, dev_in, g_imgHeight * g_imgWidth * sizeof(unsigned char));
    //导入内存
    cudaMemcpy(dev_in, g_gaussImage.data, g_imgHeight * g_imgWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    //数组导入
    cudaMemcpyToSymbol(dev_sobel_x, sobel_x, sizeof(sobel_x) );
    cudaMemcpyToSymbol(dev_sobel_y, sobel_y, sizeof(sobel_y) );
    
    //调用核函数
    //定义block thread范围
    dim3 blocks((int)((g_imgWidth+31)/32), (int)(g_imgHeight+31)/32);
    //dim3 blocks(4, 4);
    dim3 threads(16, 16);
    //单block 单thread
    sobelInCuda<<<4,4>> >(dev_out, g_imgHeight, g_imgWidth);
    
    //导出处理
    cudaMemcpy(dst_Gpu.data, dev_out, g_imgHeight * g_imgWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //停止时间点
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    //释放内存
    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_sobel_x);
    cudaFree(dev_sobel_y);


    //计算GPU所用时间
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf( "GPU对像素操作运行时间为: %.1f ms \n", elapsedTime );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    //显示处理后的图像
    cvNamedWindow("GPU处理后的图像", 0);
    resizeWindow("GPU处理后的图像", 800, 600);
    imshow("GPU处理后的图像", dst_Gpu);
}
/**
*@author：Seafood
*@name：main(int argc, char *argv[], char **env)
*@return:int
*@function：程序入口，主函数
*@para：None
*其他要注意的地方
**/
//main函数
int main(int argc, char *argv[], char **env)
{
    
    //图像读入和处理
    pictureInit();
    //Sobel算子Cpu package实现
    
    sobelCpuPackage(0,0);
    
    //测试算子
    //Sobel算子Cpu 对像素操作实现
    sobelCpuPixel();

    //Sobel算子GPU操作实现
    sobelGPUPixel();
    //结束展示
    while((char)waitKey(0) != 'q' )
    {

    }
    
    return 0;
}