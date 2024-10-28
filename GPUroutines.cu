#include "GPUroutines.cuh"
#include "processKernel.cuh"
//#include "miscfunctions.hpp"
#include <utility>
#include <vector>
#include <math.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <iostream>
#include <complex>
using namespace std;
#define CUDA_FREE_NN(x) if (x != nullptr) { cudaFree(x); x = nullptr; }

static dim3 threadsPerBlock(32, 32);
static dim3 unwrapBlocks(16, 16);
static dim3 everyPtinOneFFTBlocks;
static dim3 everyPtinOneAlineBlocks;
static dim3 everyPtinImageBlocks;
static dim3 everyPtinBatchBlocks;
static dim3 everyAlineinImageBlocks;
static dim3 everyAlineinBatchBlocks;
static dim3 everyVibZPtinOneAlineBlocks;

//vector<unsigned int> tickCount;
inline  uint64_t nextPowerOf2(uint64_t n) { // find next power of 2 for faster ffts
    uint64_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

GPU::GPU(){
	cout<<"GPU subclass initiated"<<endl;
}
void GPU::writeNum(int x){
    cout<<"write num: "<<x<<endl;
}

dim3 GPU::calcBlks(int size) {
	// this does the calculation if you want one thread for each sample point in one batch (i.e. not going to loop through sequential data points)
	int blockSize = 0;
	dim3 blks;
	float blocksNeeded = float(size) / float(threadsPerBlock.x * threadsPerBlock.y);
	if (blocksNeeded < 1) {
		blks.x = 1;
		blks.y = 1;
	}
	else if (blocksNeeded < 2) {
		blks.x = 2;
		blks.y = 1;
	}
	else {
		blocksNeeded = pow(blocksNeeded, 0.5);
		blocksNeeded = pow(2, ceil(log(blocksNeeded) / log(2)));
		blockSize = int(ceil(blocksNeeded));
		if (blockSize > 65635) {
			// this is the max number of blocks in the y dimension 
			blks.y = 65635;
			blks.x = int(ceil(float(blockSize) / 65635.0));
		}
		else {
			blks.x = blockSize;
			blks.y = blockSize;
		}
	}
	//cout << "size, calculateBlocks: "<<size<<" " << blks.x << " " << blks.y << " " << threadsPerBlock.x << " " << threadsPerBlock.y<<endl;
	return blks;
}
void GPU::linearizePDdata() {   // linearize the PD data
	// Hilbert of reference interferometer (mzi) to get k
	//cout << "GPU linearizePDdata" << endl;
	cudaError_t err;
	cufftResult result;
	//There are three separate steps because the forward and reverse FFT cannot be put into the threads (yet). Perhaps when I figure out cufftDX?
	linearize_PD1 <<<everyPtinOneFFTBlocks, threadsPerBlock >>> (d_dataIn1, d_fftHilbert, dispData.numKlinPts, alinesPerBatch);
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return;
	}
	result = cufftExecC2C(handleHilbert, d_fftHilbert, d_fftHilbert, CUFFT_FORWARD);  //first fft in Hilbert transform
	if (result != CUFFT_SUCCESS) {
		std::cerr << "cufftExecC2C error: " << result << std::endl;
		clearGPU();
		return;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return;
	}	
	linearize_PD2 <<<everyPtinOneFFTBlocks, threadsPerBlock >>> (d_fftHilbert, dispData.numKlinPts, alinesPerBatch);
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return;
	}
	result = cufftExecC2C(handleHilbert, d_fftHilbert, d_fftHilbert, CUFFT_INVERSE);  //second fft in Hilbert transform
	if (result != CUFFT_SUCCESS) {
		std::cerr << "cufftExecC2C error: " << result << std::endl;
		clearGPU();
		return;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return;
	}
	linearize_PD3 <<<everyAlineinBatchBlocks, threadsPerBlock >>> (d_dataIn, d_dataInBak, d_fftHilbert, d_k, d_klin, dispData.numKlinPts, alinesPerBatch);
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return;
	}
	return;
}
int GPU::clearGPU(){
	//cout << "clearGPU" << endl;
	cudaError_t err;
	cufftResult result;
	setupCompleted = false;
	CUDA_FREE_NN(d_fft);
	CUDA_FREE_NN(d_fftHilbert);
	CUDA_FREE_NN(d_k);
	CUDA_FREE_NN(d_klin);
	CUDA_FREE_NN(d_deltaK);
	CUDA_FREE_NN(d_dataIn);
	CUDA_FREE_NN(d_dataInBak);
	CUDA_FREE_NN(d_dataIn1);
	CUDA_FREE_NN(d_dispI);
	CUDA_FREE_NN(d_dispR);
	CUDA_FREE_NN(d_bgData);

	CUDA_FREE_NN(d_magOut);
	CUDA_FREE_NN(d_phaseOut);
	CUDA_FREE_NN(d_phaseOutVib);
	CUDA_FREE_NN(d_aveImage);
	CUDA_FREE_NN(d_aveImageCrop);
	CUDA_FREE_NN(d_viewImage);
	CUDA_FREE_NN(d_indexZPts);
	CUDA_FREE_NN(d_fftVib);
	CUDA_FREE_NN(d_fftDAQ);
	CUDA_FREE_NN(d_sigInVib);
	CUDA_FREE_NN(d_sigInDAQ);
	CUDA_FREE_NN(d_dataOut);
	CUDA_FREE_NN(d_Hanning);
	CUDA_FREE_NN(d_magFFT);
	CUDA_FREE_NN(d_fIdx);
	CUDA_FREE_NN(d_fIdxNStart);
	CUDA_FREE_NN(d_noiseRange);
	CUDA_FREE_NN(d_FFTfilter);

	//Destroy FFT Plans
	if (handle != NULL) {
		result = cufftDestroy(handle);
		if (result != CUFFT_SUCCESS) {
			std::cerr << "cufftDestroy handle error: " << result << std::endl;
			return 1;
		}
		handle = NULL;
	}
	if (handleHilbert != NULL) {
		result = cufftDestroy(handleHilbert);
		if (result != CUFFT_SUCCESS) {
			std::cerr << "cufftDestroy handlHilbert error: " << result << std::endl;
			return 1;
		}
		handleHilbert = NULL;
	}
	if (handleVib != NULL) {
		result = cufftDestroy(handleVib);
		if (result != CUFFT_SUCCESS) {
			std::cerr << "cufftDestroy handleVib error: " << result << std::endl;
			return 1;
		}
		handleVib = NULL;
	}
	if (handleDAQ != NULL) {
		result = cufftDestroy(handleDAQ);
		if (result != CUFFT_SUCCESS) {
			std::cerr << "cufftDestroy handleDAQ error: " << result << std::endl;
			return 1;
		}
		handleDAQ = NULL;
	}
	if (handleFFTfilter != NULL) {
		result = cufftDestroy(handleFFTfilter);
		if (result != CUFFT_SUCCESS) {
			std::cerr << "cufftDestroy handleFFTfilter error: " << result << std::endl;
			return 1;
		}
		handleFFTfilter = NULL;
	}
	return 0;
}
int GPU::processAScanGPU(void* dataIn, void* dataIn1, float* magOut, float* phaseOut) {
	// This function processes all Alines and returns mag and phase 
	// It can be used for any scan type, but typically for Alines or dynamic OCT Bscans
	cudaError_t err;
	cufftResult result;
	if (setupCompleted) {
		for (int b = 0; b < nBatches; ++b) {  //loop through each buffer (batch)
			int rawStartPoint = b * alinesPerBatch * dispData.numKlinPts;
			int processedStartPoint = b * alinesPerBatch * dispData.nAlinePts;
			err = cudaMemcpy(d_dataIn, (((U16*)dataIn) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cerr << "CUDA error1: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			if (protocol.performLinearization) {
				err = cudaMemcpy(d_dataInBak, (((U16*)dataIn) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
					std::cerr << "CUDA error2: " << cudaGetErrorString(err) << std::endl;
					clearGPU();
					return 1;
				}
				err = cudaMemcpy(d_dataIn1, (((U16*)dataIn1) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
					std::cerr << "CUDA error3: " << cudaGetErrorString(err) << std::endl;
					clearGPU();
					return 1;
				}
				linearizePDdata();
			}
			dispCorrAlazar <<< everyPtinOneFFTBlocks, threadsPerBlock >>> (d_dataIn, d_fft, d_dispR, d_dispI, d_bgData, dispData.numKlinPts, dispData.numKlinPts, dispData.nFFT, alinesPerBatch);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error4: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			result = cufftExecC2C(handle, d_fft, d_fft, CUFFT_FORWARD);
			if (result != CUFFT_SUCCESS) {
				std::cerr << "cufftExecC2C error5: " << result << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error6: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			processAScan1 <<< everyPtinBatchBlocks, threadsPerBlock >>> (d_fft, d_magOut, d_phaseOut, dispData.nFFT, dispData.nAlinePts, alinesPerBatch, dispData.numKlinPts);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error7: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaMemcpy(magOut + processedStartPoint, d_magOut, sizeof(float) * alinesPerBatch * dispData.nAlinePts, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cerr << "CUDA error8: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaMemcpy(phaseOut + processedStartPoint, d_phaseOut, sizeof(float) * alinesPerBatch * dispData.nAlinePts, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cerr << "CUDA error8.5: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			zeroComplex <<< everyPtinOneFFTBlocks, threadsPerBlock >>> (d_fft, alinesPerBatch * dispData.nFFT);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error9: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
		}
		return 0;
	}
	return _NOT_SETUP_F;
}
int GPU::processBScanGPU(void* dataIn, void* dataIn1, uint8_t* viewImage, float* aveImage, float* allImages, float minInt, float maxInt, int top, int bottom) {
	// this function creates one averaged Bscan, and returns the  average normalized image (float), an averaged U8 normalized image for viewing, and all the individual non-normalized Bscans
	// can be used for reg Bscans or averaged Bscans
	cudaError_t err;
	cufftResult result;
	if (setupCompleted) {
		zeroFloat <<< everyPtinImageBlocks, threadsPerBlock >>> (d_aveImage, dispData.nAlinePts * protocol.numPtsX);  // zero the average image array
		for (int b = 0; b < nBatches; ++b) {  //loop through each buffer (batch)
			//cout << "Batch,alinesPerBatch: " << b << " "<< alinesPerBatch<<endl;
			int rawStartPoint = b * alinesPerBatch * dispData.numKlinPts;
			int processedStartPoint = b * alinesPerBatch * dispData.nAlinePts;
			err = cudaMemcpy(d_dataIn, (((U16*)dataIn) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cerr << "CUDA error1: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			if (protocol.performLinearization) {
				err = cudaMemcpy(d_dataInBak, (((U16*)dataIn) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
					std::cerr << "CUDA error2: " << cudaGetErrorString(err) << std::endl;
					clearGPU();
					return 1;
				}
				err = cudaMemcpy(d_dataIn1, (((U16*)dataIn1) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
					std::cerr << "CUDA error3: " << cudaGetErrorString(err) << std::endl;
					clearGPU();
					return 1;
				}
				linearizePDdata();
			}
			dispCorrAlazar <<< everyPtinOneFFTBlocks, threadsPerBlock >>> (d_dataIn, d_fft, d_dispR, d_dispI, d_bgData, dispData.numKlinPts, dispData.numKlinPts, dispData.nFFT, alinesPerBatch);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error4: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			result = cufftExecC2C(handle, d_fft, d_fft, CUFFT_FORWARD);
			if (result != CUFFT_SUCCESS) {
				std::cerr << "cufftExecC2C error5: " << result << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error6: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			makeBScan1 <<< everyPtinBatchBlocks, threadsPerBlock >>> (d_fft, d_magOut, d_aveImage, dispData.nFFT, dispData.nAlinePts, alinesPerBatch, dispData.numKlinPts, protocol.numPtsX);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error7: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaMemcpy(allImages + processedStartPoint, d_magOut, sizeof(float) * alinesPerBatch * dispData.nAlinePts, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cerr << "CUDA error8: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			zeroComplex <<< everyPtinOneFFTBlocks, threadsPerBlock >>> (d_fft, alinesPerBatch * dispData.nFFT);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error9: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
		}
		makeBScan2 <<< everyPtinImageBlocks, threadsPerBlock >>> (d_viewImage, d_aveImage, protocol.numPtsX, dispData.nAlinePts, top, bottom, protocol.numAverages, minInt, maxInt);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			cout << top << " " << bottom << " " << minInt << " " << maxInt << " " << protocol.numPtsX << " " << dispData.nAlinePts << " " << protocol.numAverages << endl;
			std::cerr << "CUDA error10: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMemcpy(viewImage, d_viewImage, sizeof(uint8_t) * protocol.numPtsX * (bottom-top), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error11: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMemcpy(aveImage, d_aveImage, sizeof(float) * protocol.numPtsX * dispData.nAlinePts, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error12: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		return 0;
	}
	return _NOT_SETUP_F;
}
int GPU::processVolScanGPU(void* dataIn, void* dataIn1, uint8_t* viewImage, float* aveImage, float* allImages, float minInt, float maxInt, int top, int bottom) {
	// This function creates one averaged Volscan, and returns the  average normalized image (float), an averaged U8 normalized image for viewing, and all the individual non-normalized Bscans
	// can be used for reg Volscans, averaged volscans, or volume overview scans
	cudaError_t err;
	cufftResult result;
	//printf("processVolScanGPU: alinesPerBatch= %d\n",alinesPerBatch);
	//printf("DispData: %d %d %d\n", dispData.nFFT, dispData.numKlinPts, dispData.nAlinePts);
	//printf("Protocol: %d %d %d %d %d %d %d %d %d %d %d %d\n", protocol.numPtsX, protocol.numPtsY, protocol.numAverages, protocol.numdynOCT,
	//	protocol.recordsToKeepPerBuffer, protocol.buffersPerAcquisition, protocol.numSections, protocol.performLinearization, protocol.performMagScaling,
	//	protocol.minMax[0], protocol.minMax[1], protocol.numVibRepeats);
	//cout << endl;
	if (setupCompleted) {
		zeroFloat <<< everyPtinImageBlocks, threadsPerBlock >>> (d_aveImage, dispData.nAlinePts * protocol.numPtsX * protocol.numPtsY);  // zero the average image array
		for (int b = 0; b < nBatches; ++b) {  //loop through each buffer (batch)
			//printf("Batches: %d/%d  %d  %d \n", b, nBatches, protocol.numPtsX, protocol.numPtsY);
			//cout << endl;
			int rawStartPoint = b * alinesPerBatch * dispData.numKlinPts;
			int processedStartPoint = b * alinesPerBatch * dispData.nAlinePts;
			err = cudaMemcpy(d_dataIn, (((U16*)dataIn) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cerr << "CUDA error1: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			if (protocol.performLinearization) {
				err = cudaMemcpy(d_dataInBak, (((U16*)dataIn) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
					std::cerr << "CUDA error2: " << cudaGetErrorString(err) << std::endl;
					clearGPU();
					return 1;
				}
				err = cudaMemcpy(d_dataIn1, (((U16*)dataIn1) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
					std::cerr << "CUDA error3: " << cudaGetErrorString(err) << std::endl;
					clearGPU();
					return 1;
				}
				linearizePDdata();
			}
			dispCorrAlazar <<< everyPtinOneFFTBlocks, threadsPerBlock >>> (d_dataIn, d_fft, d_dispR, d_dispI, d_bgData, dispData.numKlinPts, dispData.numKlinPts, dispData.nFFT, alinesPerBatch);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error4: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			result = cufftExecC2C(handle, d_fft, d_fft, CUFFT_FORWARD);
			if (result != CUFFT_SUCCESS) {
				std::cerr << "cufftExecC2C error5: " << result << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error6: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			makeVolScan1 <<< everyPtinBatchBlocks, threadsPerBlock >>> (d_fft, d_magOut, d_aveImage, dispData.nFFT, dispData.nAlinePts, alinesPerBatch, dispData.numKlinPts, protocol.numPtsX, protocol.numPtsY, b);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "processVolScanGPU CUDA error7: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaMemcpy(allImages + processedStartPoint, d_magOut, sizeof(float) * alinesPerBatch * dispData.nAlinePts, cudaMemcpyDeviceToHost); // all Aline data
			if (err != cudaSuccess) {
				std::cerr << "CUDA error8: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			zeroComplex <<< everyPtinOneFFTBlocks, threadsPerBlock >>> (d_fft, alinesPerBatch * dispData.nFFT);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error9: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
		}
		makeVolScan2 <<< everyPtinImageBlocks, threadsPerBlock >>> (d_aveImageCrop, d_aveImage, protocol.numPtsX, protocol.numPtsY, dispData.nAlinePts, top, bottom, protocol.numAverages);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			cout << top << " " << bottom << " " << minInt << " " << maxInt << " " << protocol.numPtsX << " " << protocol.numPtsY << " " << protocol.numAverages << endl;
			std::cerr << "CUDA error10: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		makeVolScan3 <<< everyAlineinImageBlocks, threadsPerBlock >>> (d_viewImage,d_aveImageCrop, protocol.numPtsX, protocol.numPtsY, top, bottom, minInt, maxInt);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error10: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMemcpy(viewImage, d_viewImage, sizeof(uint8_t) * protocol.numPtsX * protocol.numPtsY, cudaMemcpyDeviceToHost); // en face image
		if (err != cudaSuccess) {
			std::cerr << "CUDA error11: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMemcpy(aveImage, d_aveImage, sizeof(float) * protocol.numPtsX * protocol.numPtsY* dispData.nAlinePts, cudaMemcpyDeviceToHost); // averaged full volume set
		if (err != cudaSuccess) {
			std::cerr << "CUDA error12: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		return 0;
	}
	return _NOT_SETUP_F;
}
int GPU::processMScanGPU(void* dataIn, void* dataIn1, float* magAve, float* vibOut) {
	// This function processes all Alines, averages them in to one Aline, and creates time domain phase data for later vibrational analysis 
	// Used for Mscans
	//cout << "processMScanGPU" << endl;
	cudaError_t err;
	cufftResult result= CUFFT_SUCCESS;
	if (setupCompleted) {
		for (int b = 0; b < nBatches; ++b) {  //loop through each buffer (batch)
			//cout << "batch: " << b << endl;
			int rawStartPoint = b * alinesPerBatch * dispData.numKlinPts;
			int processedStartPoint = b * alinesPerBatch * miscParam.VIBnumZPts;
			zeroComplex <<< everyPtinOneFFTBlocks, threadsPerBlock >>> (d_FFTfilter, miscParam.VIBnumZPts * miscParam.nFFTFilter);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error0.5: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaMemcpy(d_dataIn, (((U16*)dataIn) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cerr << "CUDA error1: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			if (protocol.performLinearization) {
				err = cudaMemcpy(d_dataInBak, (((U16*)dataIn) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
					std::cerr << "CUDA error2: " << cudaGetErrorString(err) << std::endl;
					clearGPU();
					return 1;
				}
				err = cudaMemcpy(d_dataIn1, (((U16*)dataIn1) + rawStartPoint), sizeof(U16) * dispData.numKlinPts * alinesPerBatch, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
					std::cerr << "CUDA error3: " << cudaGetErrorString(err) << std::endl;
					clearGPU();
					return 1;
				}
				linearizePDdata();
			}
			dispCorrAlazar <<< everyPtinOneFFTBlocks, threadsPerBlock >>> (d_dataIn, d_fft, d_dispR, d_dispI, d_bgData, dispData.numKlinPts, dispData.numKlinPts, dispData.nFFT, alinesPerBatch);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error4: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			result = cufftExecC2C(handle, d_fft, d_fft, CUFFT_FORWARD);
			if (result != CUFFT_SUCCESS) {
				std::cerr << "cufftExecC2C error5: " << result << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "processMScanGPU: CUDA error6: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			zeroFloat <<< everyPtinOneAlineBlocks, threadsPerBlock >>> (d_magOut, dispData.nAlinePts);  // zero the average image array
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "zeroFloat CUDA error6.7: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			processMScan1 <<< everyPtinOneAlineBlocks, threadsPerBlock >>> (d_fft, d_magOut, d_phaseOutVib, dispData.nFFT, dispData.nAlinePts, alinesPerBatch, dispData.numKlinPts, miscParam.VIBnumZPts, d_indexZPts);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "processMScanGPU CUDA error7: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			processMScan2 <<< everyPtinOneAlineBlocks, threadsPerBlock >>> (d_magOut, dispData.nAlinePts, alinesPerBatch);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error7.5: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaMemcpy(magAve, d_magOut, sizeof(float) * dispData.nAlinePts, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cerr << "CUDA error8: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			//cout << "alinesPerBatch, miscParam.VIBnumZPts, miscParam.nFFTFilter: " << alinesPerBatch << " " << miscParam.VIBnumZPts << " " << miscParam.nFFTFilter << endl;
			processMScan3 <<< everyVibZPtinOneAlineBlocks, threadsPerBlock >>> (d_phaseOutVib, alinesPerBatch, miscParam.VIBnumZPts,d_FFTfilter, miscParam.nFFTFilter);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error7.75: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			result = cufftExecC2C(handleFFTfilter, d_FFTfilter, d_FFTfilter, CUFFT_FORWARD);
			if (result != CUFFT_SUCCESS) {
				std::cerr << "cufftExecC2C error8: " << result << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error9: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			//cout <<"nFFT,HPfilterIdx: " << miscParam.nFFTFilter << " " << miscParam.vibHPfilterIdx << endl;
			processMScan4 <<< everyPtinBatchBlocks, threadsPerBlock >>> (miscParam.VIBnumZPts, d_FFTfilter, miscParam.nFFTFilter, miscParam.vibHPfilterIdx);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error9: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			result = cufftExecC2C(handleFFTfilter, d_FFTfilter, d_FFTfilter, CUFFT_INVERSE);
			if (result != CUFFT_SUCCESS) {
				std::cerr << "cufftExecC2C error10: " << result << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error11: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			processMScan5 <<< everyPtinBatchBlocks, threadsPerBlock >>> (d_phaseOutVib, alinesPerBatch, miscParam.VIBnumZPts, d_FFTfilter, miscParam.nFFTFilter);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error11.5: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			err = cudaMemcpy(vibOut + processedStartPoint, d_phaseOutVib, sizeof(float) * alinesPerBatch * miscParam.VIBnumZPts, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cerr << "CUDA error12: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
			zeroComplex <<< everyPtinOneFFTBlocks, threadsPerBlock >>> (d_fft, alinesPerBatch * dispData.nFFT);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "CUDA error13: " << cudaGetErrorString(err) << std::endl;
				clearGPU();
				return 1;
			}
		}
		return 0;
	}
	return _NOT_SETUP_F;
}
int GPU::processVibGPU(float* vibInput, float* vibInputAP, int currentSection, float* dataOut, float* fftMag, int* fIdx, int* fIdxNStart, int* noiseRange) {
	if (setupCompleted) {
		//cout << "processVibGPU" << endl;
		cudaError_t err;
		cufftResult result;
		int dataOutStep = 0;

		//for (int i = 0; i < 10; ++i) {
		//	int pt = miscParam.vibStartStimPt + 0 * protocol.recordsToKeepPerBuffer + i;
		//	printf("GPU vibIn: i,miscParam.vibStartStimPt, protocol.recordsToKeepPerBuffer,pt, vibInput: %d %d %d %d  %g \n",
		//		i, miscParam.vibStartStimPt, protocol.recordsToKeepPerBuffer, pt, vibInput[pt]);
		//}
		zeroComplex <<< everyAlineinBatchBlocks, threadsPerBlock >>> (d_fftVib, miscParam.VIBnumZPts * miscParam.nFFTvib*2);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error0.5: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		zeroFloat <<< 2, threadsPerBlock >>> (d_dataOut, miscParam.numSigValues);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error1: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}	
		err = cudaMemcpy(d_sigInVib, vibInput, sizeof(float) * miscParam.VIBnumZPts * protocol.recordsToKeepPerBuffer, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error2: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMemcpy(&d_sigInVib[miscParam.VIBnumZPts * protocol.recordsToKeepPerBuffer], vibInputAP, sizeof(float) * miscParam.VIBnumZPts * protocol.recordsToKeepPerBuffer, cudaMemcpyHostToDevice);  // put vib and vibAP back-to-back into the d_sigIn array
		if (err != cudaSuccess) {
			std::cerr << "CUDA error3: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}	
		processTDsignals1 <<< everyVibZPtinOneAlineBlocks, threadsPerBlock >>> (d_dataOut, d_sigInVib, protocol.recordsToKeepPerBuffer, miscParam.VIBnumZPts * 2, miscParam.vibStartStimPt, miscParam.VIBnCAPwindowPts, miscParam.nNoiseStartVib, miscParam.nNoiseVib, miscParam.numValues);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error4: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		processTDsignals2 <<< everyVibZPtinOneAlineBlocks, threadsPerBlock >>> (d_fftVib, d_sigInVib, d_Hanning, protocol.recordsToKeepPerBuffer, miscParam.VIBnumZPts * 2, miscParam.vibStartStimPt, miscParam.nStim0Vib, miscParam.nFFTvib);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error4.5: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		result = cufftExecC2C(handleVib, d_fftVib, d_fftVib, CUFFT_FORWARD);
		if (result != CUFFT_SUCCESS) {
			std::cerr << "cufftExecC2C error4: " << result << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error5: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		for (int i = 0; i < protocol.numFreqToAnalyze; ++i) {
			noiseRange[i] = int(ceil(float(miscParam.endIdxVib) / 10.0));
			fIdx[i] = int(round(protocol.freqArray[i][currentSection / protocol.numLevel] * double(miscParam.nFFTvib) / protocol.laserTriggerRate));
			fIdxNStart[i] = int(ceil((miscParam.endIdxVib - fIdx[i]) / 2 + fIdx[i]));
			if ((fIdxNStart[i] + noiseRange[i]) > miscParam.endIdxVib) { noiseRange[i] = miscParam.endIdxVib - fIdxNStart[i]; }
			//cout << "fIDx " << fIdx[i] <<" "<< protocol.numFreqToAnalyze<< endl;
		}
		err = cudaMemcpy(d_fIdx, &fIdx[0], sizeof(int) * protocol.numFreqToAnalyze, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "processVibGPU CUDA error6: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMemcpy(d_fIdxNStart, &fIdxNStart[0], sizeof(int) * protocol.numFreqToAnalyze, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "processVibGPU CUDA error7: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMemcpy(d_noiseRange, &noiseRange[0], sizeof(int) * protocol.numFreqToAnalyze, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error8: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		float fft_corr = 2.0 * 2.0 / float(miscParam.nStim0Vib);
		processFFTsignals1 <<< everyVibZPtinOneAlineBlocks, threadsPerBlock >>> (d_fftVib, miscParam.VIBnumZPts * 2, miscParam.endIdxVib, miscParam.nFFTvib,d_magFFT, fft_corr);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error9: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
	
		//cout << "getting ready for processFFTsignals2: " << miscParam.VIBnumZPts * 2 * protocol.numFreqToAnalyze << endl;
		//cout << "getting ready for processFFTsignals2: " << miscParam.endIdxVib << " "<< miscParam.nFFTvib<<" "<<miscParam.numValues << endl;
		processFFTsignals2 <<< 1, threadsPerBlock >>> (d_dataOut, d_fftVib, miscParam.VIBnumZPts * 2, miscParam.endIdxVib, protocol.numFreqToAnalyze, d_fIdx, miscParam.nFFTvib, miscParam.numValues, d_magFFT);
		//cout << "back from processFFTsignals2: " << miscParam.VIBnumZPts * 2 * protocol.numFreqToAnalyze << endl;
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error9.3: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		processFFTsignals3 <<< everyVibZPtinOneAlineBlocks, threadsPerBlock >>> (d_dataOut, miscParam.VIBnumZPts * 2, miscParam.endIdxVib, protocol.numFreqToAnalyze, d_fIdxNStart, d_noiseRange, miscParam.numValues, d_magFFT);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error9.6: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMemcpy(fftMag, d_magFFT, sizeof(float) * miscParam.endIdxVib, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error10: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMemcpy(dataOut, d_dataOut, sizeof(float) * miscParam.numValues * miscParam.VIBnumZPts * 2, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error11: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		
		
		//for (int z = 0;z < miscParam.VIBnumZPts; ++z) {
		//	cout << "Analysis zPt: " <<z<< endl;
		//	for (int i = 0; i < miscParam.numValues;++i) {
		//		cout << "z,i,dataOut[i]: " << z<<" "<<i << " " << dataOut[i+z* miscParam.numValues] << endl;
		//	}
		//}	
		
		return 0;
	}
	return _NOT_SETUP_F;
}
void GPU::recalculateBatches() {
	// minimize the number of batches processed by the GPU to speed things up. However, if batchSize gets too big the GPU crashes. 
	//cout << "recalculateBatches" << endl;
	double maxBatchSizeMemory = 5e8;
	double totalNumberAlines = int(protocol.buffersPerAcquisition * protocol.recordsToKeepPerBuffer / protocol.numSections);
	double newBatches = 1;
	double newAlinesPerBatch = totalNumberAlines / newBatches;
	double batchSizeMemory = newAlinesPerBatch * dispData.nFFT;
	//cout << "original batches: " << protocol.recordsToKeepPerBuffer << " " << nBatches << " " << dispData.nFFT << " " << batchSizeMemory << endl;
	//cout << "recalculated batches: " << newAlinesPerBatch << " " << newBatches << " " << newBatches * newAlinesPerBatch * dispData.nFFT << endl;
	while (batchSizeMemory > maxBatchSizeMemory) {
		newBatches++; //add another batch
		newAlinesPerBatch = totalNumberAlines / newBatches;
		//cout << "outerloop: " << newBatches << endl;
		while (int(totalNumberAlines) % int(newBatches) != 0) {
			newBatches++;
			//cout << "innerloop: " << newBatches << endl;
		} // keep adding batches until the number of Alines is evenly divisible by the number of batches (so there are no "extra" Alines)
		newAlinesPerBatch = totalNumberAlines / newBatches;
		batchSizeMemory = newAlinesPerBatch * dispData.nFFT;
	}
	//cout << "original batches: " << alinesPerBatch << " " << nBatches << " " << fftLength << " " << batchSizeMemory << endl;
	//cout << "recalculated batches: " << newAlinesPerBatch << " " << newBatches << " " << fftLength << " " << newAlinesPerBatch * fftLength << endl;
	alinesPerBatch = newAlinesPerBatch;
	nBatches = newBatches;
}
int GPU::setupGPUDAQVib(int* misc, double* micResponseF, double* micResponseDB,double* freqArray, double* levelArray, int* indexZPts, double* timeInput,double* ch0TimeArray, double* ch1TimeArray, double* repTimeArray){
	//cout << "setupGPUDAQVib" << endl;

	cudaError_t err;
	cufftResult result;
	setupCompleted = false;

	miscParam.processMic = misc[0];
	miscParam.processVib= misc[1];
	miscParam.DAQnumPtsIn= misc[2];
	miscParam.DAQnCAPwindowPts= misc[3];
	miscParam.VIBnumZPts= misc[4];
	miscParam.VIBnumPtsIn= misc[5];  // this is for the entire freq/level series (not just one buffer, which is protocol.recordsToKeepPerBuffer).
	miscParam.VIBnCAPwindowPts= misc[6];
	miscParam.micResponseF_size= misc[7];

	vector<double> emptyVec(alinesPerBatch);
	miscParam.timeInput = emptyVec;
	vector<double> emptyVecLevel(protocol.numLevel, 0);
	vector<double> emptyVecFreq(protocol.numFreq, 0);
	vector<vector<double>> emptyVec1(protocol.numFreq, emptyVecLevel);
	vector<vector<double>> emptyVec2(protocol.numFreqToAnalyze, emptyVecFreq);
	vector<vector<double>> emptyVec3(2, emptyVecLevel);
	miscParam.ch0TimeArray = emptyVec1;
	miscParam.ch1TimeArray = emptyVec1;
	miscParam.repTimeArray = emptyVec1;
	protocol.freqArray = emptyVec2;
	protocol.levelArray = emptyVec3;
	//cout << "gothere1 " << protocol.numFreq<<" "<< protocol.numFreqToAnalyze<<" "<< protocol.numLevel<< endl;
	everyVibZPtinOneAlineBlocks = calcBlks(miscParam.VIBnumZPts);

	miscParam.numSigs = max(miscParam.VIBnumZPts * 2, 3);  // max number of sigs to be analyzed
	miscParam.numValues = (3 + protocol.numFreqToAnalyze * 4); // number of values in each dataOut for each sig;
	miscParam.numSigValues = miscParam.numSigs * miscParam.numValues; //number of values to store for each sig

	//printf("miscParam.numSigs,protocol.numValues,miscParam.numSigValues: %d %d %d\n", miscParam.numSigs, miscParam.numValues, miscParam.numSigValues);

	miscParam.nStim0Vib = floor(protocol.stimTime0 * protocol.laserTriggerRate);
	//printf("miscParam.nStim0Vib,protocol.stimTime0, protocol.laserTriggerRate: %d %g %g \n", miscParam.nStim0Vib, protocol.stimTime0, protocol.laserTriggerRate);
	miscParam.vibStartStimPt = floor(protocol.stimDelay0 * protocol.laserTriggerRate);
	miscParam.nNoiseStartVib = floor(double(protocol.recordsToKeepPerBuffer) / 2.0);  // calculate noise floor from the last half of the rep (This will only work if there is no stimulus present at this time)
	if (miscParam.nNoiseStartVib < miscParam.vibStartStimPt + miscParam.nStim0Vib) { miscParam.nNoiseStartVib = miscParam.vibStartStimPt + miscParam.nStim0Vib; }
	if (miscParam.nNoiseStartVib < miscParam.vibStartStimPt + miscParam.VIBnCAPwindowPts) { miscParam.nNoiseStartVib = miscParam.vibStartStimPt + miscParam.VIBnCAPwindowPts; }
	if (miscParam.nNoiseStartVib >= protocol.recordsToKeepPerBuffer -4) { miscParam.nNoiseStartVib = protocol.recordsToKeepPerBuffer - 4; }
	miscParam.nNoiseVib = protocol.recordsToKeepPerBuffer - miscParam.nNoiseStartVib;

	//printf("miscParam.nStim0Vib,miscParam.vibStartStimPt,miscParam.nNoiseStartVib,miscParam.nNoiseVib,protocol.recordsToKeepPerBuffer %d %d %d %d %d \n",
	//	miscParam.nStim0Vib, miscParam.vibStartStimPt, miscParam.nNoiseStartVib, miscParam.nNoiseVib, protocol.recordsToKeepPerBuffer);

	miscParam.nStim0DAQ = floor(protocol.stimTime0 * protocol.samplingRateIn);
	miscParam.DAQStartStimPt = floor(protocol.stimDelay0 * protocol.samplingRateIn);
	miscParam.nNoiseStartDAQ = floor(double(miscParam.DAQnumPtsIn) / 2.0);  // calculate noise floor from the last half of the rep (This will only work if there is no stimulus present at this time)
	if (miscParam.nNoiseStartDAQ < miscParam.DAQStartStimPt + miscParam.nStim0DAQ) { miscParam.nNoiseStartDAQ = miscParam.DAQStartStimPt + miscParam.nStim0DAQ; }
	if (miscParam.nNoiseStartDAQ < miscParam.DAQStartStimPt + miscParam.DAQnCAPwindowPts) { miscParam.nNoiseStartDAQ = miscParam.DAQStartStimPt + miscParam.DAQnCAPwindowPts; }
	if (miscParam.nNoiseStartDAQ >= miscParam.DAQnumPtsIn-4) { miscParam.nNoiseStartDAQ = miscParam.DAQnumPtsIn - 4; }
	miscParam.nNoiseDAQ = miscParam.DAQnumPtsIn - miscParam.nNoiseStartDAQ;

	miscParam.nFFTvib = protocol.numfftptsVib;  // use a power of 2 with a lot of points
	miscParam.nFFTDAQ = nextPowerOf2(miscParam.nStim0DAQ) << 3;
	miscParam.nFFTFilter = nextPowerOf2(protocol.recordsToKeepPerBuffer) << 3;  // used to filter out low freq noise from each phase vib signal

	miscParam.endIdxVib = protocol.endIdxVib;
	miscParam.endIdxDAQ = floor(miscParam.nFFTDAQ / 2) + 1;
	miscParam.endIdxFilter = floor(miscParam.nFFTFilter / 2) + 1;
	miscParam.vibHPfilterIdx = floor(miscParam.nFFTFilter * protocol.VibHighPassFilter); 
	miscParam.nBinsPeakSearchVib = int(floor(protocol.deltaFreqPeakSearch * miscParam.nFFTvib / protocol.laserTriggerRate));
	miscParam.nBinsPeakSearchDAQ = int(floor(protocol.deltaFreqPeakSearch * miscParam.nFFTDAQ / protocol.samplingRateIn));

	//cout << "With GPU-- numfftpts,numInputSamples: " << miscParam.nFFTvib << " " << miscParam.nStim0Vib << endl;
	for (int i = 0; i < alinesPerBatch; ++i) {
		miscParam.timeInput[i] = timeInput[i];
	}
	int p1 = 0;
	int p2 = 0;
	for (int f = 0; f < protocol.numFreq; ++f) {
		for (int i1 = 0; i1 < protocol.numFreqToAnalyze; ++i1) {
			protocol.freqArray[i1][f] = freqArray[p1];
			//cout << "freq: " << f << " " << i1 << " " << protocol.freqArray[i1][f] << endl;
			p1++;
		}
		for (int l = 0; l < protocol.numLevel; ++l) {
			miscParam.ch0TimeArray[f][l]= ch0TimeArray[p2];
			miscParam.ch1TimeArray[f][l]= ch1TimeArray[p2];
			miscParam.repTimeArray[f][l]= repTimeArray[p2];
			//cout << "time: " << f << " " << l << " " << miscParam.repTimeArray[f][l]<<" "<< miscParam.ch0TimeArray[f][l]<<" "<< miscParam.ch1TimeArray[f][l] << endl;
			p2++;
		}
	}
	//cout << "gothere2 " << protocol.numLevel<<endl;
	int p3 = 0;
	for (int l = 0; l < protocol.numLevel; ++l) {
		protocol.levelArray[0][l]= *(levelArray+p3);
		p3++;
		protocol.levelArray[1][l]= *(levelArray+p3);
		p3++;
		//cout << "level: " << l << " " << protocol.levelArray[0][l] << " " << protocol.levelArray[1][l] <<  endl;
	}
	//cout << "gothere3 " << miscParam.VIBnumZPts<< endl;
	miscParam.indexZPts = {};
	for (int i = 0; i < miscParam.VIBnumZPts; ++i) {
		miscParam.indexZPts.push_back(indexZPts[i]);
	}
	//cout << "gothere4 " << miscParam.micResponseF_size<<endl;
	miscParam.micResponseF = {};
	miscParam.micResponseDB = {};
	for (int i = 0; i < miscParam.micResponseF_size; ++i) {
		miscParam.micResponseF.push_back(micResponseF[i]);
		miscParam.micResponseDB.push_back(micResponseDB[i]);
	}
	//cout << "gothere4.25 " << endl;

	

	vector<float> Hanning(miscParam.nStim0Vib);
	for (int i = 0; i < miscParam.nStim0Vib; ++i) {
		Hanning[i] = 0.5 * (1.0 - cos(2.0 * M_PI * float(i) / float(miscParam.nStim0Vib)));
	}
	//cout << "gothere4.5 " << miscParam.nStim0Vib<< endl;
	err = cudaMalloc(&d_Hanning, sizeof(float) * miscParam.nStim0Vib);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	//cout << "gothere4.75 " << miscParam.nStim0Vib << endl;
	err = cudaMemcpy(d_Hanning, &Hanning[0], sizeof(float) * miscParam.nStim0Vib, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	//cout << "gothere5 " << endl;
	err = cudaMalloc(&d_indexZPts, sizeof(int) * miscParam.VIBnumZPts);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMemcpy(d_indexZPts, &miscParam.indexZPts[0], sizeof(int) * miscParam.VIBnumZPts, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_fIdx, sizeof(int) * protocol.numFreqToAnalyze);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_fIdxNStart, sizeof(int) * protocol.numFreqToAnalyze);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_noiseRange, sizeof(int) * protocol.numFreqToAnalyze);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	//cout << "dispData.nAlinePts:" << dispData.nAlinePts<< endl;
	err = cudaMalloc(&d_magOut, sizeof(float) * dispData.nAlinePts);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_phaseOutVib, sizeof(float) * protocol.recordsToKeepPerBuffer * miscParam.VIBnumZPts);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_sigInVib, sizeof(float) * protocol.recordsToKeepPerBuffer * miscParam.VIBnumZPts * 2); //vib and vibAP
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_sigInDAQ, sizeof(float) * miscParam.DAQnumPtsIn * 3); // mic, CAP, CM
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	int sizeFFTmagArray = max(miscParam.endIdxVib, miscParam.endIdxDAQ);
	err = cudaMalloc(&d_magFFT, sizeof(float) * sizeFFTmagArray * miscParam.numSigs);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	//cout << "d_dataOut size: " << miscParam.numSigValues << endl;
	err = cudaMalloc(&d_dataOut, sizeof(float) * miscParam.numSigValues); 
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_FFTfilter, sizeof(Complex) * miscParam.VIBnumZPts * miscParam.nFFTFilter);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	result = cufftPlan1d(&handleFFTfilter, miscParam.nFFTFilter, CUFFT_C2C, miscParam.VIBnumZPts);
	if (result != CUFFT_SUCCESS) {
		std::cerr << "cufftPlan1d error: " << result << std::endl;
		clearGPU();
		return 1;
	}
	//cout << "d_fftVib size " << miscParam.VIBnumZPts * miscParam.nFFTvib * 2 << " " << miscParam.VIBnumZPts << " " << miscParam.nFFTvib << endl;
	err = cudaMalloc(&d_fftVib, sizeof(Complex) * miscParam.VIBnumZPts * miscParam.nFFTvib*2);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_fftDAQ, sizeof(Complex) * 3 * miscParam.nFFTDAQ);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	//cout << "gothere7 " << endl;
	result = cufftPlan1d(&handleVib, miscParam.nFFTvib, CUFFT_C2C, miscParam.VIBnumZPts*2);
	if (result != CUFFT_SUCCESS) {
		std::cerr << "cufftPlan1d error: " << result << std::endl;
		clearGPU();
		return 1;
	}
	result = cufftPlan1d(&handleDAQ, miscParam.nFFTDAQ, CUFFT_C2C, 3);
	if (result != CUFFT_SUCCESS) {
		std::cerr << "cufftPlan1d error: " << result << std::endl;
		clearGPU();
		return 1;
	}
	setupCompleted = true;
	//cout << "setupGPUDAQVib completed" << endl;
	return 0;
}
int GPU::setupGPU(int* protocolParam, double* protocolParamD, int* dispDataParam, float* dispRe, float* dispIm, float* background){
	//cout << "setupGPU" << endl;
	cudaError_t err;
	cufftResult result;
	dispData.nFFT = dispDataParam[0];
	dispData.numKlinPts = dispDataParam[1];
	dispData.nAlinePts = dispDataParam[2];
	//dispData.nAlinePts = 40;
	//printf("DispData: %d %d %d\n", dispData.nFFT, dispData.numKlinPts, dispData.nAlinePts);

	protocol.numPtsX = protocolParam[0];
	protocol.numPtsY = protocolParam[1];
	protocol.numAverages = protocolParam[2];
	protocol.numdynOCT = protocolParam[3];
	protocol.recordsToKeepPerBuffer = protocolParam[4];
	protocol.buffersPerAcquisition = protocolParam[5];
	protocol.numSections = protocolParam[6];
	protocol.performLinearization = protocolParam[7];
	protocol.performMagScaling = protocolParam[8];
	protocol.numVibRepeats = protocolParam[9];
	protocol.scanType = ScanType(protocolParam[10]);
	protocol.sizeSectionsDAQIn = protocolParam[11];
	protocol.numFreqToAnalyze = protocolParam[12];
	protocol.numFreq = protocolParam[13];
	protocol.numLevel = protocolParam[14];
	protocol.numfftptsVib = protocolParam[15];
	protocol.endIdxVib = protocolParam[16];

	protocol.minMax[0] = float(protocolParamD[0]);
	protocol.minMax[1] = float(protocolParamD[1]);
	protocol.samplingRateIn = protocolParamD[2];
	protocol.repTime = protocolParamD[3];
	protocol.stimTime0 = protocolParamD[4];
	protocol.stimDelay0 = protocolParamD[5];
	protocol.laserTriggerRate = protocolParamD[6];
	protocol.mic_VoltsPerPascal = protocolParamD[7];
	protocol.mic_gain = protocolParamD[8];
	protocol.VibHighPassFilter= protocolParamD[9];
	protocol.deltaFreqPeakSearch = protocolParamD[10];

	//cout << getScanTypeString(protocol.scanType) << " " << protocol.scanType << endl;
	//printf("Protocol: %d %d %d %d %d %d %d %d %d %d %d %d\n", protocol.numPtsX, protocol.numPtsY, protocol.numAverages, protocol.numdynOCT,
   //protocol.recordsToKeepPerBuffer, protocol.buffersPerAcquisition, protocol.numSections, protocol.performLinearization, protocol.performMagScaling,
   //	protocol.minMax[0], protocol.minMax[1], protocol.numVibRepeats);

	int x = clearGPU(); 
	if (x != cudaSuccess){
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error spot 1: " << cudaGetErrorString(err) << std::endl;
		}
		return _GPU_CLEAR_ERROR;
	}
	//cout << "Back from clearGPU" << endl;
	int kernelSize;
	setupCompleted = false;

	alinesPerBatch = protocol.recordsToKeepPerBuffer;
	nBatches = int(protocol.buffersPerAcquisition / protocol.numSections);
	//cout << "Original alinesPerBatch,nBatches: " << alinesPerBatch << " " << nBatches << endl;
	//if (protocol.scanType == ASCAN) {
	//	recalculateBatches();  //this can reduce the number of batches (and thus save time), but won't work when averaging or for most protocols
		//cout << "Revised alinesPerBatch,nBatches: " << alinesPerBatch << " " << nBatches << endl;
	//}

	kernelSize = dispData.numKlinPts * protocol.recordsToKeepPerBuffer;
	int error = 0;
	everyPtinOneFFTBlocks = calcBlks(dispData.nFFT);
	everyPtinOneAlineBlocks = calcBlks(dispData.nAlinePts);
	everyPtinImageBlocks = calcBlks(dispData.nAlinePts * protocol.numPtsX * protocol.numPtsY);
	everyPtinBatchBlocks = calcBlks(dispData.nAlinePts * alinesPerBatch); 
	everyAlineinImageBlocks = calcBlks(protocol.numPtsX * protocol.numPtsY); 
	everyAlineinBatchBlocks = calcBlks(alinesPerBatch); 
	

	int totalAlines = alinesPerBatch * nBatches;
	int totalRawData = totalAlines * dispData.numKlinPts;
	int totalFFTData = totalAlines * dispData.nFFT;

	// Special memory allocation for each different scantype
	if (protocol.scanType == BSCAN) {
		err = cudaMalloc(&d_magOut, sizeof(float) * dispData.nAlinePts * alinesPerBatch);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_aveImage, sizeof(float) * dispData.nAlinePts * protocol.numPtsX);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_viewImage, sizeof(uint8_t) * dispData.nAlinePts * protocol.numPtsX);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
	}
	if ((protocol.scanType == VOLSCAN)||(protocol.scanType == VOLOVERVIEWSCAN)) {
		err = cudaMalloc(&d_magOut, sizeof(float) * dispData.nAlinePts * alinesPerBatch);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_aveImage, sizeof(float) * dispData.nAlinePts * protocol.numPtsX * protocol.numPtsY);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_aveImageCrop, sizeof(float) * dispData.nAlinePts * protocol.numPtsX * protocol.numPtsY);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_viewImage, sizeof(uint8_t) * protocol.numPtsX*protocol.numPtsY);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
	}
	else if ((protocol.scanType == ASCAN) || (protocol.scanType == DYNBSCAN) || (protocol.scanType == NEMOSCAN)) {
		err = cudaMalloc(&d_magOut, sizeof(float) * dispData.nAlinePts * alinesPerBatch);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_phaseOut, sizeof(float) * dispData.nAlinePts * alinesPerBatch);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
	}


	// General memory allocation for all scantypes
	err = cudaMalloc(&d_dataIn, sizeof(U16) * dispData.numKlinPts * alinesPerBatch);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_fft, sizeof(Complex) * alinesPerBatch * dispData.nFFT);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_dispR, sizeof(float) * dispData.numKlinPts);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_dispI, sizeof(float) * dispData.numKlinPts);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMalloc(&d_bgData, sizeof(float) * dispData.numKlinPts);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	zeroComplex <<<everyPtinOneFFTBlocks, threadsPerBlock >>> (d_fft, alinesPerBatch * dispData.nFFT);
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMemcpy(d_dispR,&dispRe[0], sizeof(float) * dispData.numKlinPts, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMemcpy(d_dispI, &dispIm[0], sizeof(float) * dispData.numKlinPts, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	err = cudaMemcpy(d_bgData, &background[0], sizeof(float) * dispData.numKlinPts, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		clearGPU();
		return 1;
	}
	result = cufftPlan1d(&handle, dispData.nFFT, CUFFT_C2C, alinesPerBatch);
	if (result != CUFFT_SUCCESS) {
		std::cerr << "cufftPlan1d error: " << result << std::endl;
		clearGPU();
		return 1;
	}
	result = cufftPlan1d(&handleHilbert, dispData.numKlinPts, CUFFT_C2C, alinesPerBatch);
	if (result != CUFFT_SUCCESS) {
		std::cerr << "cufftPlan1d error: " << result << std::endl;
		clearGPU();
		return 1;
	}
	if (protocol.performLinearization) {  // only allocate this memory if linearization is needed
		err = cudaMalloc(&d_dataIn1, sizeof(U16) * dispData.numKlinPts * alinesPerBatch);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_dataInBak, sizeof(U16) * dispData.numKlinPts * alinesPerBatch);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_fftHilbert, sizeof(Complex) * alinesPerBatch * dispData.numKlinPts);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_k, sizeof(float) * alinesPerBatch * dispData.numKlinPts);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_klin, sizeof(float) * alinesPerBatch * dispData.numKlinPts);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
		err = cudaMalloc(&d_deltaK, sizeof(float) * alinesPerBatch);
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			clearGPU();
			return 1;
		}
	}
	setupCompleted = true;
	//cout << "finished setupGPU" << endl;
	return _SUCCESS;
}
int GPU::resetGPU() {
	cudaError_t err = cudaDeviceReset();
	if (err != cudaSuccess) {
		cout << "!!!!!!!!!!!! error resetting the GPU:  " << err << endl;
		return err;
	}
	return 0;
}
