#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
	public:
	int M, N, O;

	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int M, int N, int O);

	~Layer();

	void setOutput(float *data);
	void clear();
	void bp_clear();
};


// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]);
__global__ void fp_bias_c1(float preact[6][24][24], float bias[6]);
__global__ void fp_preact_s1(float input[6][24][24], float preact[6][12][12], float weight[1][2][2]);
__global__ void fp_bias_s1(float preact[6][12][12], float bias[1]);
__global__ void fp_preact_c2(float input[6][12][12], float preact[16][8][8], float weight[16][6][5][5]);
__global__ void fp_bias_c2(float preact[16][8][8], float bias[16]);
__global__ void fp_preact_s2(float input[16][8][8], float preact[16][4][4], float weight[1][2][2]);
__global__ void fp_bias_s2(float preact[16][4][4], float bias[1]);
__global__ void fp_preact_c3(float input[16][4][4], float preact[120], float weight[120][16][4][4]);
__global__ void fp_bias_c3(float preact[120], float bias[120]);
__global__ void fp_preact_f1(float input[120], float preact[84], float weight[84][120]);
__global__ void fp_bias_f1(float preact[84], float bias[84]);
__global__ void fp_preact_f2(float input[84], float preact[10], float weight[10][84]);
__global__ void fp_bias_f2(float preact[10], float bias[10]);

// Back propagation kernels
__global__ void bp_weight_f2(float d_weight[10][84], float d_preact[10], float p_output[84]);
__global__ void bp_bias_f2(float bias[10], float d_preact[10]);

__global__ void bp_output_f1(float d_output[84], float n_weight[10][84], float nd_preact[10]);
__global__ void bp_preact_f1(float d_preact[84], float d_output[84], float preact[84]);
__global__ void bp_weight_f1(float d_weight[84][120], float d_preact[84], float p_output[120]);
__global__ void bp_bias_f1(float bias[84], float d_preact[84]);

__global__ void bp_output_c3(float d_output[120], float n_weight[84][120], float nd_preact[84]);
__global__ void bp_preact_c3(float d_preact[120], float d_output[120], float preact[120]);
__global__ void bp_weight_c3(float d_weight[120][16][4][4], float d_preact[120], float p_output[16][4][4]);
__global__ void bp_bias_c3(float bias[120], float d_preact[120]);

__global__ void bp_output_s2(float d_output[16][4][4], float n_weight[120][16][4][4], float nd_preact[120]);
__global__ void bp_preact_s2(float d_preact[16][4][4], float d_output[16][4][4], float preact[16][4][4]);
__global__ void bp_weight_s2(float d_weight[1][2][2], float d_preact[16][4][4], float p_output[16][8][8]);
__global__ void bp_bias_s2(float bias[1], float d_preact[16][4][4]);

__global__ void bp_output_c2(float d_output[16][8][8], float n_weight[1][2][2], float nd_preact[16][4][4]);
__global__ void bp_preact_c2(float d_preact[16][8][8], float d_output[16][8][8], float preact[16][8][8]);
__global__ void bp_weight_c2(float d_weight[16][6][5][5], float d_preact[16][8][8], float p_output[6][12][12]);
__global__ void bp_bias_c2(float bias[16], float d_preact[16][8][8]);

__global__ void bp_output_s1(float d_output[6][12][12], float n_weight[16][6][5][5], float nd_preact[16][8][8]);
__global__ void bp_preact_s1(float d_preact[6][12][12], float d_output[6][12][12], float preact[6][12][12]);
__global__ void bp_weight_s1(float d_weight[1][2][2], float d_preact[6][12][12], float p_output[6][24][24]);
__global__ void bp_bias_s1(float bias[1], float d_preact[6][12][12]);

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][2][2], float nd_preact[6][12][12]);
__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]);




/*
__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]);
__global__ void bp_bias_f(float bias[10], float d_preact[10]);
__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]);
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24]);
__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6]);

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]);
__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]);
*/
