#include "processKernel.cuh"
#define _USE_MATH_DEFINES
#include <math.h>
#include <cufft.h>
#include <cuComplex.h>
#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>
#include <cstdint>
#include <iostream>
// CUDA runtime
#include <cuda_runtime.h>

#define EXTERN extern "C"
using namespace std;
typedef uint16_t U16;
typedef cuFloatComplex Complex;
typedef uint64_t U64;
typedef int16_t I16;
typedef int64_t I64;

__device__ int getTid(){
	return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ int getTid2D2D(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.y) + threadIdx.x;
	return threadId;
}
__device__ int getTid2D1D(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}
__device__ int getTid3D2D(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}
__device__ int getTid3D1D(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}
__device__ float carg(Complex z){ //Complex number you want the phase of
	return atan2f(z.y, z.x);// polar angle
}
__device__ float magNorm16(float mag,float normLow,float normHigh) {
	if (mag < normLow) mag = normLow;
	if (mag > normHigh) mag = normHigh;
	mag = UINT16_MAX * (mag - normLow) / (normHigh - normLow); // convert to 16-bit
	return mag;
}
__device__ uint8_t magNorm8(float mag,float normLow,float normHigh){
	if (mag < normLow) mag = normLow;
	if (mag > normHigh) mag = normHigh;
	uint8_t out = uint8_t(UINT8_MAX * (mag - normLow) / (normHigh - normLow)); // convert to 8-bit
	return out;
}
__global__ void linearize_PD1(U16* d_dataIn1, Complex* d_fftHilbert, int nPts, int alinesPerBatch){
	//determine the current thread ID
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < alinesPerBatch * nPts; i += stride) {
		d_fftHilbert[i].x = float(d_dataIn1[i]);  // no filtering of the MZI data as a test
		d_fftHilbert[i].y = 0.0;
		//if (i < 5) {
		//	printf("linearize_PD1: %d %d : %d : %f %f \n",tid, i, d_dataIn1[i], d_fftHilbert[i].x, d_fftHilbert[i].y);
		//}
	}
}
__global__ void linearize_PD2(Complex* d_fftHilbert, int nPts, int alinesPerBatch){
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	//printf("%d  %d %d  %d %d \n",tid, blockDim.x, blockDim.y,gridDim.x,gridDim.y);
	for (int i = tid; i < alinesPerBatch * nPts; i += stride) {
		int rawIndex = i % nPts;
		//int currLine = i / nPts;
		int hN = nPts >> 1;  // half of the length (nPts/2)
		//if (i < 5) {
		//	printf("linearize_PD2 B: %d %d : %f %f \n", tid, i, d_fftHilbert[i].x, d_fftHilbert[i].y);
		//}
		//printf("linearize_PD2 B: %d %d", (nPts % 2), (nPts % 2 == 1));
		if (rawIndex == 0) {
			d_fftHilbert[i].x = 0;
			d_fftHilbert[i].y = 0;
		} else {
			if (rawIndex < hN) { // multiply the appropriate values by 2 (those that should be multiplied by 1 are left intact)
				d_fftHilbert[i].x *= 2;
				d_fftHilbert[i].y *= 2;
			} else if (rawIndex > hN) {  // set the remaining values to 0
				d_fftHilbert[i].x = 0;
				d_fftHilbert[i].y = 0;
			} else if (rawIndex == hN && nPts % 2 == 1) { // If it's odd and >1, the middle value must be multiplied by 2
				d_fftHilbert[i].x *= 2;
				d_fftHilbert[i].y *= 2;
			}
		}
		//if (i < 5) {
		//	printf("linearize_PD2 A: %d %d : %f %f \n", tid, i, d_fftHilbert[i].x, d_fftHilbert[i].y);
		//}
	}
}
__global__ void linearize_PD3(U16* d_dataIn, U16* d_dataInBak, Complex* d_fftHilbert, float* d_k, float* d_klin, int nPts, int alinesPerBatch){
	//determine the current thread ID
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	//printf("linearize_PD3: %d %d : %d %d : %d %d : %d \n",tid, stride, blockDim.x, blockDim.y,gridDim.x,gridDim.y, alinesPerBatch);
	for (int i = tid; i < alinesPerBatch; i += stride) {
		int startPt = i * nPts;
		//printf("---------------------------linearize_PD3 1: %d %d %d \n", tid, i, startPt);
		for (int i1 = 0; i1 < nPts; ++i1) {
			//if (i1 < 5) {
			//	printf("linearize_PD3 1: %d %d %d : %f %f \n", tid, i, i1, d_fftHilbert[i1].x, d_fftHilbert[i1].y);
			//}
			int p = i1 + startPt;
			d_fftHilbert[p].x /= nPts;
			d_fftHilbert[p].y /= nPts;
			d_k[p] = atan2(d_fftHilbert[p].y, d_fftHilbert[p].x);   // calculate the angle
			if (i1 > 1) {
				float d = d_k[p] - d_k[p - 1];   // unwrap it
				if (d > M_PI) { d_k[p] -= 2 * M_PI * (1 + floor(abs(d) / (2 * M_PI))); }
				else if (d < -M_PI) { d_k[p] += 2 * M_PI * (1 + floor(abs(d) / (2 * M_PI))); }
			}
		}
		//printf("---------------------------linearize_PD3 2: %d %d %d \n", tid, i, startPt);
		// create the klin array
		float k0 = d_k[startPt];
		float deltaK = (d_k[nPts * (tid + 1) - 1] - k0) / nPts;
		//printf("linearize_PD3 2: %d %d : %f %f \n", tid, i, k0, deltaK);
		//U16 temp[16384];
		for (int i1 = 0; i1 < nPts; ++i1) {
			int p = i1 + startPt;
			d_klin[p] = k0 + i1 * deltaK;
			//temp[i1] = d_dataIn[p];  //make a temp array of the pd data for this sweep
			//if (i1 < 5) {
			//	printf("linearize_PD3 3: %d %d %d : %f %d \n", tid, i, p, d_klin[p], temp[i1]);
			//}
		}
		// interpolate dataIn along klin
		//printf("---------------------------linearize_PD3 3: %d %d %d \n", tid, i, startPt);
		int iTemp = 0;
		for (int i1 = 0; i1 < nPts; ++i1) {
			int p = i1 + startPt;
           
			if (d_klin[p] >= d_k[startPt + nPts - 2]) {
				iTemp = nPts - 2;  // special case: beyond right end 
			} else {
				while(d_klin[p] < d_k[iTemp + startPt]) {
					iTemp--;
					printf("----decreasing iTemp: %d %d %d : %f %f", tid, i1, iTemp, d_klin[p], d_k[iTemp + startPt]);
				}  // may need to restart iterating from the beginning 
				while (d_klin[p] >= d_k[iTemp + startPt + 1]) {
					iTemp++; 
					//printf("Increasing iTemp: %d %d %d : %f %f", tid, i1, iTemp, d_klin[p], d_k[iTemp + startPt + 1]);
				}  // find left end of interval for interpolation
			}
			float xL = d_k[iTemp + startPt], yL = d_dataInBak[iTemp + startPt], xR = d_k[iTemp + startPt + 1], yR = d_dataInBak[iTemp + startPt + 1];      // points on either side (unless beyond ends)
			if (d_klin[p] < xL) { yR = yL; }
			if (d_klin[p] > xR) { yL = yR; }
			float dydx = (yR - yL) / (xR - xL);       // gradient
			d_dataIn[p] = U16(yL + dydx * (d_klin[p] - xL));  // write over dataIn buffer with linearized PD data
			//if (i1 < 6) {
			//	float dOut = yL + dydx * (d_klin[p] - xL);
			//	printf("linearize_PD3 4: %d %d %d : %f : %f %f %f %f : %f %d \n", tid, i1, iTemp, d_klin[p], xL, xR, yL, yR, dOut, d_dataIn[p]);
			//}
		}
		//printf("---------------------------linearize_PD3 4: %d %d %d \n", tid, i, startPt);
	}
}
__global__ void processAScan1(Complex* input, float* magOut, float* phaseOut, int fftL, int nAlinePts, int alinesPerBatch, int rawSize){
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nAlinePts * alinesPerBatch; i += stride) {
		int zPt = i % nAlinePts;
		int aLine = int(i / nAlinePts);
		int pt = (aLine * fftL) + zPt;
		float temp1 = cuCabsf(input[pt]) / rawSize;
		magOut[i] = log10f(temp1 + 1);
		phaseOut[i]= carg(input[pt]);
	}
}
__global__ void processMScan1(Complex* input, float* magOut, float* phaseOut, int fftL, int nAlinePts, int alinesPerBatch, int rawSize, int VIBnumZPts, int* indexZPts)
{
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nAlinePts * alinesPerBatch; i += stride) {
		int zPt = i % nAlinePts;
		int aLine = int(i / nAlinePts);
		int ptFFT = (aLine * fftL) + zPt;
		float temp1 = cuCabsf(input[ptFFT]) / rawSize;
		magOut[zPt] += log10f(temp1 + 1);  // add it to the running sum to get an averaged Aline
		if ((zPt >= indexZPts[0]) && (zPt <= indexZPts[VIBnumZPts - 1])) { // if it is a z point we want to analyze
			phaseOut[(zPt-indexZPts[0])*alinesPerBatch + aLine] = float(carg(input[ptFFT]));
			//if ((zPt - indexZPts[0]) == 1) {
			//	printf("i,zPt,indexZPts[0],vibSpot,vib: %d %d %d %d  %g \n", i, zPt, indexZPts[0], (zPt - indexZPts[0]) * alinesPerBatch + aLine, float(carg(input[ptFFT])));
			//}
		}
	}
}
__global__ void processMScan2(float* magOut, int nAlinePts, int alinesPerBatch)  // average the single Aline
{
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nAlinePts; i += stride) {
		magOut[i] /= float(alinesPerBatch);  // get an averaged Aline
	}
}
__global__ void processMScan3(float* phaseOut, int alinesPerBatch, int VIBnumZPts, Complex* d_FFTfilter, int nFFT)  // unwrap the time domain tracing for each zPt
{
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < VIBnumZPts; i += stride) {
		float startPoint=0;
		float prevPt=0;
		for (int p = 0; p < alinesPerBatch; ++p) {
			float currPt = phaseOut[p + i * alinesPerBatch];
			float shift=0;
			if (p == 0) {
				startPoint = currPt;
			}
			else {
				float d = currPt - prevPt + shift;
				while (d > M_PI) {
					shift = -2 * M_PI;
					d += shift;
				}
				while (d < -M_PI) {
					shift = 2 * M_PI;
					d += shift;
				}
			}
			currPt += shift;
			prevPt = currPt;
			phaseOut[p + i * alinesPerBatch] = currPt - startPoint; // raw phase out signal
			d_FFTfilter[i * nFFT + p].x = currPt - startPoint; // phase put into FFT array for later filtering
			//if (i==1){
			//	if ((p > 99) && (p < 110)) {
			//		printf("processMscan with GPU prefilter: p,z,vibInput: %d %d %g\n", p, i, phaseOut[p + i * alinesPerBatch]);
			//	}
			//}
		}
	}
}
__global__ void processMScan4(int VIBnumZPts, Complex* d_FFTfilter, int nFFT, int HPFIdx) // perform high pass filtering
{
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < VIBnumZPts * HPFIdx; i += stride) {
		int zPt = i / HPFIdx;
		int fftBin = i % HPFIdx;
		int ptFFT = (zPt * nFFT) + fftBin;
		d_FFTfilter[ptFFT].x = 0; // remove any freq information in these low freq bins
		d_FFTfilter[ptFFT].y = 0;
	}
}
__global__ void processMScan5(float* phaseOut, int alinesPerBatch, int VIBnumZPts, Complex* d_FFTfilter, int nFFT)  // copy filtered signal back into phaseOut array
{
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < VIBnumZPts * alinesPerBatch; i += stride) {
		int zPt = i / alinesPerBatch;
		int aLine = i % alinesPerBatch;
		int ptFFT = (zPt * nFFT) + aLine;
		phaseOut[i] = d_FFTfilter[ptFFT].x / nFFT;
	}
}
__global__ void test(int x,int y) {
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < x; i += stride) {
		printf("test: %d %d \n", x,y);
	}
}
__global__ void processTDsignals1(float* dataOut, float* sigIn, int nPts, int nSigs, int startPt, int nCAPwindowPts, int nNoiseStart, int nNoise, int nSkip)
{// // calculate peak-to-peak CAP signal
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nSigs; i += stride) {
		int sigStart = i * nPts;
		//printf("processTDsignals: %d %d %d %d %d \n", i, nPts, nSigs, sigStart, stride);
		float maxPP = 0;
		float minPP = 0;
		for (int p = startPt; p < startPt + nCAPwindowPts; ++p) {
			if (maxPP < sigIn[sigStart + p]) { maxPP = sigIn[sigStart + p]; }
			if (minPP > sigIn[sigStart + p]) { minPP = sigIn[sigStart + p]; }
		}
		dataOut[i*nSkip] = maxPP - minPP; // peak-to-peak CAP signal
		float sumsq = 0;
		float variance = 0;
		float t = sigIn[sigStart + nNoiseStart];
		for (int p = 0; p < nNoise; ++p) {
			sumsq += (sigIn[sigStart + nNoiseStart + p] * sigIn[sigStart + nNoiseStart + p]);
			//printf("i,p,sigIn[sigStart+nNoiseStart+p],sumsq: %d %d  %g %g\n", i, p, sigIn[sigStart + nNoiseStart + p], sumsq);
			if (p > 0) {
				t += sigIn[sigStart + nNoiseStart + p];
				float diff = (((p + 1)) * sigIn[sigStart + nNoiseStart + p]) - t;
				variance += (diff * diff) / ((p + 1.0) * p);
			}
		}
		//printf("i, nNoise,nNoiseStart, sumsq, variance:  %d %d %d  %g %g\n", i, nNoise, sumsq, variance);
		dataOut[i * nSkip + 1] = sqrt(sumsq / float(nNoise));  // RMS of the noise
		dataOut[i * nSkip + 2] = sqrt(variance / float(nNoise - 1)); // SD of the noise
	}
}
__global__ void processTDsignals2(Complex* d_fft, float* sigIn, float* d_Hanning, int nPts, int nSigs, int startPt, int nStim0, int nFFT)
{  // copy the portion of the time domain curve during stim0 into the array to do the FFT
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nSigs * nStim0; i += stride) {
		int tdPt = i % nStim0;
		int sig = int(i / nStim0);
		int fftPt = (sig * nFFT) + tdPt;
		Complex tmp;
		tmp.x = float(sigIn[startPt+ sig * nPts + tdPt] * d_Hanning[tdPt]);
		tmp.y = 0;
		d_fft[fftPt] = tmp;
		//if (i < 10) {printf("GPU: i,startPt,sig,nPts,tdPt,startPt + sig * nPts + tdPt, sigIn,tmp: %d %d %d %d %d %d  %g %g\n", i,startPt,sig,nPts,tdPt, startPt + sig * nPts + tdPt, sigIn[startPt + sig * nPts + tdPt], tmp.x);}
	}
}
__global__ void processFFTsignals1(Complex* d_fft, int nSigs,int endIdx, int nFFT, float* d_magFFT, float fft_corr)
{ // calculate the fft magnitude for the entire spectrum
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < endIdx * nSigs; i += stride) {
		int sig = i / endIdx;
		int fftPt = i % endIdx;
		d_magFFT[i] = fft_corr * cuCabsf(d_fft[sig * nFFT + fftPt]);;
		//if (fftPt < 10) { printf("fftPt,sig,d_magFFT: %d %d %f\n", fftPt, sig, d_magFFT[i]); }
	}
}
__global__ void processFFTsignals2(float* dataOut, Complex* d_fft, int nSigs,int endIdx, int numFreqToAnalyze, int* d_fIdx, int nFFT, int nSkip, float* d_magFFT)
{ // select the mag and phase from the FFT for the chosen frequencies
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nSigs * numFreqToAnalyze; i += stride) {
		int sig = i / numFreqToAnalyze;
		int f = i % numFreqToAnalyze;
		dataOut[i * nSkip + 3 + f * 4 + 0] = d_magFFT[sig * endIdx + d_fIdx[f]];  //mag at the peak
		dataOut[i * nSkip + 3 + f * 4 + 1] = carg(d_fft[sig * nFFT + d_fIdx[f]]);  //phase 
	}
}
__global__ void processFFTsignals3(float* dataOut, int nSigs,int endIdx, int numFreqToAnalyze, int* d_fIdxNStart, int* d_noiseRange, int nSkip,float* d_magFFT)
{ // calculate ave noise and SD from the FFT
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nSigs * numFreqToAnalyze; i += stride) {
		int sig = i / numFreqToAnalyze;
		int sigStart = sig * endIdx;
		int f = i % numFreqToAnalyze;
		float variance = 0;
		float t = d_magFFT[sigStart + d_fIdxNStart[f]];
		for (int p = 1; p < d_noiseRange[f]; ++p) {
			t += d_magFFT[sigStart + d_fIdxNStart[f]+p];
			float diff = (((p + 1)) * d_magFFT[sigStart + d_fIdxNStart[f] + p]) - t;
			variance += (diff * diff) / ((p + 1.0) * p);
		}
		dataOut[sig * nSkip + 3 + f * 4 + 2] = t / float(d_noiseRange[f]);  // average mag of the noise
		dataOut[sig * nSkip + 3 + f * 4 + 3] = sqrt(variance / float(d_noiseRange[f] - 1)); // SD of the noise			
	}
}
__global__ void makeBScan1(Complex* input, float* magOut, float* aveImage, int fftL, int nAlinePts, int alinesPerBatch,int rawSize, int nX)
{
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nAlinePts * alinesPerBatch; i += stride) {
		//printf("tid, i, stride: %d %d %d \n", tid, i, stride);
		int zPt = i % nAlinePts;
		int aLine = int(i / nAlinePts);
		int pt = (aLine * fftL) + zPt;
		float temp1 = cuCabsf(input[pt])/rawSize;
		float temp = log10f(temp1 + 1);
		magOut[i] = temp;  // the full Bscan dataset
		int aveImageAline = aLine % nX;
		aveImage[aveImageAline* nAlinePts+zPt] += temp;  // add this to the averaged Bscan image
		//if (i < 5) { printf("makeBScan1: %d %3.2f %3.2f %3.2f %3.2f\n", i, input[pt].x, input[pt].y, temp1, temp); }
	}
}
__global__ void makeBScan2(uint8_t* viewImage,float* aveImage,int nX, int nAlinePts, int top, int bottom,int nAve, float minInt, float maxInt)
{
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nAlinePts * nX; i += stride) {
		if (nAve > 1) { aveImage[i] /= nAve; }
		int zPt = i % nAlinePts;
		int aLine = int(i / nAlinePts);
		if ((zPt >= top) && (zPt < bottom)) {  // if pixel is within the area to view, put it into the image
			viewImage[aLine * (bottom-top)+zPt - top] = magNorm8(aveImage[i], minInt, maxInt);
		}
	}
}
__global__ void makeVolScan1(Complex* input, float* magOut, float* aveImage, int fftL, int nAlinePts, int alinesPerBatch, int rawSize, int nX,int nY, int b)
{
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nAlinePts * alinesPerBatch; i += stride) {
		int zPt = i % nAlinePts;
		int aLine = int(i / nAlinePts);
		int pt = (aLine * fftL) + zPt;
		float temp1 = cuCabsf(input[pt]) / rawSize;
		float temp = log10f(temp1 + 1);
		magOut[i] = temp;  // the full VolScan dataset
		//int aveImageAline = b*nX + aLine%nX; // one b-scan per buffer
		int slice = b % nY;
		aveImage[slice*nX*nAlinePts + aLine * nAlinePts + zPt] += temp;
		//aveImage[aveImageAline * nAlinePts + zPt] += temp;  // add this to the averaged VolScan image
		//if (i < 5) { printf("makeVolScan1: %d %3.2f %3.2f %3.2f %3.2f\n", i, input[pt].x, input[pt].y, temp1, temp); }
	}
}
__global__ void makeVolScan2(float* aveImageCrop, float* aveImage, int nX, int nY, int nAlinePts, int top, int bottom, int nAve)
{
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nAlinePts * nX * nY; i += stride) {
		if (nAve > 1) { aveImage[i] /= nAve; }
		int zPt = i % nAlinePts;
		int aLine = int(i / nAlinePts);
		if ((zPt >= top) && (zPt < bottom)) {  // if pixel is within the area to view, put it into the image
			aveImageCrop[aLine * (bottom - top) + zPt - top] = aveImage[i];
		}
	}
}
__global__ void makeVolScan3(uint8_t* viewImage, float* aveImageCrop, int nX, int nY, int top, int bottom, float minInt, float maxInt)
{
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < nX * nY; i += stride) {
		viewImage[i] = 0;;
		for (int z = 0; z < bottom - top; ++z) {
			viewImage[i] += magNorm8(aveImageCrop[i * (bottom - top) + z], minInt, maxInt);
		}
	}
}
__global__ void dispCorrAlazar(U16* in, Complex* out, float* dispReal, float* dispImag,float* bgData, int rawSize, int validSize, int nFFT, int numAlines)
{
	//determine the current thread ID
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y* gridDim.x * gridDim.y;
	//printf("%d  %d %d  %d %d \n",tid, blockDim.x, blockDim.y,gridDim.x,gridDim.y);
	for (int i = tid; i < numAlines * validSize;i+=stride) {
		int rawIndex = i % rawSize;
		int currLine = i / rawSize;
		int outputIndex = currLine * nFFT + rawIndex;
		float val = float(in[i]) - bgData[rawIndex];	// Subtract background
		//printf("%d - %d - %d %d \n", tid, i, validIndex, outputIndex);
		Complex tmp;
		//Dispersion Compensation and windowing
		tmp.x = val * dispReal[rawIndex];
		tmp.y = val * dispImag[rawIndex];
		//Write the value to the correct location in the output array
		//if (tid < 5) {printf("dispCorrAlazar: %d:  %d %3.2f  %3.2f %3.2f \n", tid, in[i], val, tmp.x, tmp.y);}
		out[outputIndex] = tmp;
	}
}
__global__ void zeroComplex(Complex* array, int length) //length of the array
{
	//determine the current thread ID
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	//printf("%d  %d %d  %d %d \n",tid, blockDim.x, blockDim.y,gridDim.x,gridDim.y);
	for (int i = tid; i < length; i += stride) {
		array[i].x = 0;
		array[i].y = 0;
	}
}
__global__ void zeroFloat(float* array,int length) {
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < length; i += stride) {
		array[tid] = 0;
	}
}
__global__ void zeroDouble(double* array, int length) {
	int tid = getTid2D2D();
	int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	for (int i = tid; i < length; i += stride) {
		array[tid] = 0;
	}
}
__global__ void zeroU16(U16* array,	int length)//length of the array
{
	int tid = getTid2D2D();
	if (tid < length) {
		array[tid] = 0;
	}
}
__global__ void zeroU64(U64* array,	int length)//length of the array
{
	int tid = getTid2D2D();
	if (tid < length) {
		array[tid] = 0;
	}
}

