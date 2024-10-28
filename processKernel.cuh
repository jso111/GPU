#include <cstdint>
#include <math.h>
#include <cufft.h>
#include <cuComplex.h>
#include <stdio.h>
#include <cstdlib>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
typedef uint16_t U16;
typedef cuFloatComplex Complex;
typedef uint64_t U64;
typedef int16_t I16;
typedef int64_t I64;
#define EXTERN extern "C"
using namespace std;
typedef cuFloatComplex Complex;

// Max value in range for scaling images to 16-bit
//tatic constexpr unsigned int kImageRangeMax16bit = UINT16_MAX;
//static constexpr unsigned int kImageRangeMax8bit = UINT8_MAX;
//static constexpr unsigned long long kImageRangeMax16bitULL = (unsigned long long) kImageRangeMax16bit;

__device__ int getTid();
__device__ int getTid2D2D();
__device__ int getTid2D1D();
__device__ int getTid3D2D();
__device__ int getTid3D1D();
__device__ float carg(Complex z); 
__device__ float magNorm16(float mag,float normLow, float normHigh); 
__global__ void linearize_PD1(U16* d_dataIn1, Complex* d_fftHilbert, int rawSize, int alinesPerBatch);
__global__ void linearize_PD2(Complex* d_fftHilbert, int rawSize, int alinesPerBatch);
__global__ void linearize_PD3(U16* d_dataIn, U16* d_dataInBak, Complex* d_fftHilbert, float* d_k, float* d_klin, int nPts, int alinesPerBatch);
__global__ void processAScan1(Complex* input, float* magOut, float* phaseOut, int fftL, int nAlinePts, int alinesPerBatch, int rawSize);
__global__ void makeBScan1(Complex* input, float* magOut, float* aveImage, int fftL, int nAlinePts, int alinesPerBatch, int rawSize, int nX);
__global__ void makeBScan2(uint8_t* viewImage, float* aveImage, int nX, int nAlinePts, int top, int bottom, int nAve, float minInt, float maxInt);
__global__ void makeVolScan1(Complex* input, float* magOut, float* aveImage, int fftL, int nAlinePts, int alinesPerBatch, int rawSize, int nX, int nY, int b);
__global__ void makeVolScan2(float* aveImageCrop, float* aveImage, int nX, int nY, int nAlinePts, int top, int bottom, int nAve);
__global__ void makeVolScan3(uint8_t* viewImage, float* aveImageCrop, int nX, int nY, int top, int bottom, float minInt, float maxInt);
__global__ void processMScan1(Complex* input, float* magOut, float* phaseOut, int fftL, int nAlinePts, int alinesPerBatch, int rawSize, int VIBnumZPts, int* indexZPts);
__global__ void processMScan2(float* magOut, int nAlinePts, int alinesPerBatch);
__global__ void processMScan3(float* phaseOut, int alinesPerBatch, int VIBnumZPts, Complex* d_FFTfilter, int nFFT);
__global__ void processMScan4(int VIBnumZPts, Complex* d_FFTfilter, int nFFT, int HPFIdx);
__global__ void processMScan5(float* phaseOut, int alinesPerBatch, int VIBnumZPts, Complex* d_FFTfilter, int nFFT);
__global__ void processTDsignals1(float* dataOut, float* sigIn, int nPts, int nSigs, int startPt, int nCAPwindowPts, int nNoiseStart, int nNoise, int nSkip);
__global__ void processTDsignals2(Complex* d_fft, float* sigIn, float* d_Hanning, int nPts, int nSigs, int startPt, int nStim0, int nFFT);
__global__ void processFFTsignals1(Complex* d_fft, int nSigs, int endIdx, int nFFT, float* d_magFFT, float fft_corr);
__global__ void processFFTsignals2(float* dataOut, Complex* d_fft, int nSigs, int endIdx, int numFreqToAnalyze, int* d_fIdx, int nFFT, int nSkip, float* d_magFFT);
__global__ void processFFTsignals3(float* dataOut, int nSigs, int endIdx, int numFreqToAnalyze, int* d_fIdxNStart, int* d_noiseRange, int nSkip, float* d_magFFT);


__global__ void test(int x, int y);
__global__ void dispCorrAlazar(U16* in, //raw input data
	Complex* out, //output to FFT
	float* dispReal, //real part of dispersion compensation
	float* dispImag, //imagingary part of dispersion compensatio
	float* bgData,   //background data
	int rawSize, //size of raw dataset
	int validSize, //size of dataset with invalid points removed
	int nFFT, //size of zero padded FFT
	int numAlines); //number of alines per FFT batch
__global__ void zeroComplex(Complex* array, //array to zero
	int length); //length of the array
__global__ void zeroFloat(float* array,int length);
__global__ void zeroU16(U16* array,//array to zero
	int length);//length of the array
__global__ void zeroU64(U64* array, //array to zero
	int length);//length of the array
__global__ void zeroDouble(double* array, int length);