#pragma once
#include <math.h>
#include <cufft.h>
#include <cuComplex.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include "processKernel.cuh"
//#include "DataStructures.h"

#define EXTERN extern "C"
using namespace std;
typedef uint16_t U16;
//typedef cuFloatComplex Complex;
typedef uint64_t U64;
typedef int16_t I16;
typedef int64_t I64;

/*****************
Overhead Functions
*****************/

#define _DATA_U16 0
#define _DATA_PACKED_U64 1

//Error Codes
#define _SUCCESS 0
#define _GPU_CLEAR_ERROR 0
#define _ZERO_FLOAT_F -151
#define _ZERO_COMPLEX_F -152
#define _ZERO_U16_F -153
#define _ZERO_U64_F - 154
#define _INTERFEROGRAM_F -155
#define _DISP_CORR_ALA_F -156
#define _ALINE_FFT_F -157
#define _CROP_MAGNITUDE_F -158
#define _AVG_MAG_TIME_PHASE_F -159
#define _AVG_MAG_AND_PREP_PHASE_F -160
#define _M_SCAN_F -161
#define _PHASE_MAG_AND_PHASE_PHASE_F -162
#define _AVG_MAG_PHASE_POINT_F -163
#define _CROP_MAG_PHASE_F -164
#define _EXTRACT_NOISE_F -165
#define _CALCULATE_MASKED_PHASE_F -166
#define _CALCULATE_SPATIALLY_AVERAGED_PHASE_F -167

#define _MEM_ALLOCATION_F -225
#define _MEMCPY_TO_DEVICE_F -226
#define _FFT_PLAN_F -227
#define _MEMCPY_TO_HOST_F -228
#define _LINEARIZEPD1_F -300
#define _LINEARIZEPD2_F -301
#define _LINEARIZEPDDATA_F -302

#define _NOT_SETUP_F -404
#define _NOT_INTERFEROGRAM_SETUP_F -405
#define _NOT_M_SCAN_SETUP_F -406
#define _INVALID_FUNCTION -408
#define _INVALID_DATA_TYPE_F -409
#define _GPU_IN_ERROR_STATE_F -410
#define _KERNEL_TOO_LARGE_F -411
#define _INVALID_MINMAX_F -412
#define _INVALID_CROP_RANGE -413

enum ScanType {
    NONE=0,
    ASCAN=1,
    BSCAN=2,
    DYNBSCAN=3,
    VOLSCAN=4,
    VOLOVERVIEWSCAN=5,
    MSCAN=6,
    BMSCAN=7,
    VOLMSCAN=8,
    DISPERSION=9,
    BACKGROUND=10,
    FREERUN=11,
    TESTCAL=12,
    ABR=13,
    CAP=14,
    CM=15,
    DPOAE=16,
    VSEP=17,
    EP=18,
    NEMOSCAN=19
};
struct experimentalProtocol {
    // General
    enum ScanType scanType;
    double samplingRateOut;
    double samplingRateIn;
    double laserTriggerRate;
    bool saveRaw;

    // from soundStim
    vector<double> sp0Volts;
    vector<double> sp1Volts;
    vector<vector<uint8_t>> digitalOut;
    double repTime; // Repetition time (s)
    double stimTime0; // Stimulus time (s)
    double stimDelay0; // delay after start of the repetition before the sound stimulus starts
    double stimTime1; // Stimulus time (s)
    double stimDelay1; // delay after start of the repetition before the sound stimulus starts
    bool alternatePolarity; //if you want sound stim reps to alternate polarity
    vector<vector<double>> freqArray; // array of frequencies to produce sound (Hz) speaker1
    vector<vector<double>> levelArray; // array of levels to produce sound (dB SPL) speaker 1
    int numFreq; // number of stimulation freqs in a protocol
    int numLevel;
    int numFreqToAnalyze; // number of freqs to analyze for each stimulation freq
    double timeMax;
    int numPtsOut;
    int numPtsIn;
    int numRampPts;
    int numAttenShiftPoints=34;
    vector<vector<double>> ch0TimeArray;  //start times for each stimulus freq/level combination (numFreq, numLevel)
    vector<vector<double>> ch1TimeArray;
    vector<vector<double>> repTimeArray;

    //from scanMirror
    vector<double> mirrorXvolts;
    vector<double> mirrorYvolts;
    vector<double> mirrorXPtsMM;
    vector<double> mirrorYPtsMM;
    int numPtsX; // num discrete points in the X direction
    int numPtsY; // num discrete points in the Y direction
    int numVibRepeats; // num times to repeatedly collect each x/y location for vib scan
    int nAlines;  // number per buffer
    int numdynOCT;  // number per buffer
    int X=0; // step in the number of X points
    int Y=0; // step in the number of Y points
    int N=0; // step in the number of vib repetitions for each X/Y location
    double timePerPoint; //how long should mirror be fixed at each point:0 for imaging scans; the sound stimulus repetition time for mScans
    int numPtsFlyback;
    int numRepeat=1; // when doing a scan, number of times each A line is repeated to keep the scan rate slow enough for the mirrors
    vector<double> vertexAline;
    vector<vector<double>> vertexBMScan;
    double zPixelDimMM;

    //from either, depending upon the protocol type
    int buffersPerAcquisition=0;
    int recordsPerBuffer=0;
    int recordsToKeepPerBuffer=0;
    bool moveMirrorsLong=true;
    int samplesPerRecord=0;
    int numKlinPts=0;
    int nAlinePts=0;
    int numPtsRep=0;
    int numAverages;  // number of times to repeat the sound or scan and average the responses
    int sizeOfRaw; //size of the octData.rawCh0 datatype
    int numSections; // if a scan is too long, collect it in smaller sections (1 or more buffers) and then paste them back together
    int currentSection; // current section being collected
    int sizeSectionsDAQIn; // length of each section
    int sizeSectionsDAQOut; // length of each section
    int sizeSectionsOCT; // length of each section
    bool createAveData=false;  // make true before averaging DAQ and Vib data
    int rawDataLength;
    int AlineDataLength;
    string window; // type of window used for dispersion correction (Hann seems to work best)
    double maxImageIntensity_Bscan;
    double maxImageIntensity_Overview;
    double minImageIntensity_Bscan;
    double minImageIntensity_Overview;
    float minMax[2]; // 2 point array from config file of "standard" scaling setup
    bool performLinearization; // does laser data need to be linearized?
    bool performMagScaling;  // should Aline mag data be scaled to U8?
    double mic_VoltsPerPascal;
    double mic_gain;
    double VibHighPassFilter;
    double deltaFreqPeakSearch;
    double CAPwindow;
    int numfftptsDAQ;
    int numfftptsVib;
    int endIdxDAQ;
    int endIdxVib;
    double estimatedTime;
    uint32_t tickCountStart;
    bool zigZagScan; // for volScan
    int numShiftPts; // for volScan zigZag
    bool processMic;
    bool processVib;
    int maxInt; //intensity of image taking sliders into account
    int minInt;
    int top; // top and bottom of image taking sliders into account
    int bottom;
    vector<double> micResponseF;
    vector<double> micResponseDB;
};
struct DispersionData {
    int startSample;
    int endSample;
    int numKlinPts;
    int nFFT;
    int nAlinePts;
    int idx0;
    int idx1;
    double PDfilterCutoffLow;
    double PDfilterCutoffHigh;
    double BKGfilter;
    double MZIfilter;
    double magWinfilter;
    bool useDV; //whether or not to use DVV file to crop raw PD data and linearize the signal. It should not be used until dispersion is corrected and a DVV file exists
    string window; // type of window used in dispersion (Hann may be the best)
    
    vector<float> magWin;
    vector<float> phaseCorr;
    vector<float> background;
    vector<float> dispRe;
    vector<float> dispIm;
    vector<float> uncorrAline;
    vector<float> corrAline;
    vector<float> k;
    vector<float> klin;
    vector<U16> linearizedPD;
    vector<U16> rawCh0;
    vector<U16> rawCh1;
};
struct miscParameters {  // used by GPU
    int processMic;
    int processVib;
    int DAQnumPtsIn;
    int DAQnCAPwindowPts;
    int VIBnumZPts;
    int VIBnumPtsIn;
    int VIBnCAPwindowPts;
    int micResponseF_size;
    int nFFTvib;
    int nFFTDAQ;
    int nFFTFilter;
    int nStim0Vib;
    int vibStartStimPt;
    int nNoiseStartVib;
    int nNoiseVib;
    int nStim0DAQ;
    int DAQStartStimPt;
    int nNoiseStartDAQ;
    int nNoiseDAQ;
    int numSigs;
    int numValues;
    int numSigValues;
    int endIdxVib;
    int endIdxDAQ;
    int endIdxFilter;
    int vibHPfilterIdx;  //high pass filter IDX for the vib signal
    int nBinsPeakSearchVib;
    int nBinsPeakSearchDAQ;
    vector<double> timeInput;
    vector<vector<double>> ch0TimeArray;  //start times for each stimulus freq/level combination (numFreq, numLevel)
    vector<vector<double>> ch1TimeArray;
    vector<vector<double>> repTimeArray;
    vector<int> indexZPts;
    vector<double> micResponseF;
    vector<double> micResponseDB;
};



class GPU
{
public:
    GPU();
    int clearGPU();
    void writeNum(int x);
    int processAScanGPU(void* dataIn, void* dataIn1, float* magOut, float* phaseOut);
    int processBScanGPU(void* dataIn, void* dataIn1, uint8_t* viewImage, float* aveImage, float* allImages, float minInt, float maxInt, int top, int bottom);
    int processVolScanGPU(void* dataIn, void* dataIn1, uint8_t* viewImage, float* aveImage, float* allImages, float minInt, float maxInt, int top, int bottom);
    int processMScanGPU(void* dataIn, void* dataIn1, float* magAve, float* vibIn);
    int processVibGPU(float* vibInput, float* vibInputAP, int currentSection, float* dataOut, float* fftMag, int* fIdx, int* fIdxNStart, int* noiseRange);
    int setupGPUDAQVib(int* misc, double* micResponseF, double* micResponseDB,double* freqArray, double* levelArray, int* indexZPts, double* timeInput,double* ch0TimeArray, double* ch1TimeArray, double* repTimeArray);
    int setupGPU(int* protocolParam, double* protocolParamD, int* dispDataParam, float* dispRe, float* dispIm, float* background);
    void recalculateBatches();
    int resetGPU();

private:
    dim3 calcBlks(int size);
    void linearizePDdata();

    int alinesPerBatch; //number of a-lines to process at once
    int nBatches; //Number of batches (max = 1 currently)
    bool setupCompleted; //Boolean for ensuring proper functionality

    Complex* d_fft = nullptr; //Dataset for the FFT
    Complex* d_fftHilbert = nullptr; //Dataset for the FFT for the Hilbert Transform
    float* d_k = nullptr; //Dataset for k
    float* d_klin = nullptr; //Dataset for klin
    float* d_deltaK = nullptr; 
    U16* d_dataIn = nullptr;
    U16* d_dataInBak = nullptr; // used for linearization for phase unwrapping
    U16* d_dataIn1 = nullptr;  // Ch1 data if needed for linearization
    float* d_dispI = nullptr; //Save imaginary Dispersion data
    float* d_dispR = nullptr; //Save the real dispersion data
    float* d_bgData = nullptr; //Save the background data

    float* d_magOut = nullptr; //Output for the magnitude
    float* d_phaseOut = nullptr; //Output for the phase
    float* d_phaseOutVib = nullptr; //Output for the phase
    float* d_aveImage = nullptr; //output for the averageImage
    float* d_aveImageCrop = nullptr; //output for the averageImage
    uint8_t* d_viewImage = nullptr; //output for the scaled image
    int* d_indexZPts = nullptr; // array of z points to analyze
    Complex* d_fftVib = nullptr; //Dataset for the FFT
    Complex* d_fftDAQ = nullptr; //Dataset for the FFT
    float* d_sigInVib = nullptr; //vib signals
    float* d_sigInDAQ = nullptr; //DAQ signals
    float* d_dataOut = nullptr; //Output for the phase
    float* d_Hanning = nullptr; //Output for the phase
    float* d_magFFT= nullptr; //mag of the time domain vibration signal FFT
    int* d_fIdx = nullptr; // array of z points to analyze
    int* d_fIdxNStart = nullptr; // array of z points to analyze
    int* d_noiseRange = nullptr; // array of z points to analyze
    Complex* d_FFTfilter = nullptr; //filter the phase signal to remove low freq components

    cufftHandle handle = NULL; //Plan for the a-line FFT
    cufftHandle handleHilbert = NULL; //Plan for the Hilbert transform FFT
    cufftHandle handleDAQ = NULL; //Plan for the daq transform FFT
    cufftHandle handleVib = NULL; //Plan for the vib transform FFT
    cufftHandle handleFFTfilter = NULL; //Plan for the vib filter transform FFT

    // The structures work, but I can't pass them from the QT code to this code correctly
    // I guess the items are not templated exactly the same during the two compiling processes
    // I'm not sure why, but for now, I will just pass the necessary items individually, and then put them back into the correct structures.
    experimentalProtocol protocol;
    DispersionData dispData;
    miscParameters miscParam;
};
