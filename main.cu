#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_s1 = Layer(2*2, 1, 12*12*6);
static Layer l_c2 = Layer(5*5*6, 16, 8*8*16);
static Layer l_s2 = Layer(2*2, 1, 4*4*16);
static Layer l_c3 = Layer(4*4*16, 120, 1*1*120);
static Layer l_f1 = Layer(120, 84, 84);
static Layer l_f2 = Layer(84, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char **argv)
{
	srand(time(NULL));

	/*CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return 1;
	}
	*/
	loaddata();
	learn();
	test();

	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}
	//fprintf(stdout ,"Init Forward\n");
	l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_c2.clear();
	l_s2.clear();
	l_c3.clear();
	l_f1.clear();
	l_f2.clear();
	//fprintf(stdout ,"Init Done\n");
	clock_t start, end;
	start = clock();

	l_input.setOutput((float *)input);
	fprintf(stdout ,"Conv1 Forwarding\n");
	fp_preact_c1<<<128, 128>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
	fp_bias_c1<<<128, 128>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
	apply_step_function<<<128, 128>>>(l_c1.preact, l_c1.output, l_c1.O);

	fprintf(stdout ,"Pool1 Forwarding\n");
	fp_preact_s1<<<128, 8>>>((float (*)[24][24])l_c1.output, (float (*)[12][12])l_s1.preact, (float (*)[2][2])l_s1.weight);
	fp_bias_s1<<<128, 128>>>((float (*)[12][12])l_s1.preact, l_s1.bias);
	apply_step_function<<<128, 128>>>(l_s1.preact, l_s1.output, l_s1.O);

	fprintf(stdout ,"Conv2 Forwarding\n");
	fp_preact_c2<<<128, 128>>>((float (*)[12][12])l_s1.output, (float (*)[8][8])l_c2.preact, (float (*)[6][5][5])l_c2.weight);
	fp_bias_c2<<<128, 128>>>((float (*)[8][8])l_c2.preact, l_c2.bias);
	apply_step_function<<<128, 128>>>(l_c2.preact, l_c2.output, l_c2.O);

	fprintf(stdout ,"Pool2 Forwarding\n");
	fp_preact_s2<<<128, 128>>>((float (*)[8][8])l_c2.output, (float (*)[4][4])l_s2.preact, (float (*)[2][2])l_s2.weight);
	fp_bias_s2<<<128, 128>>>((float (*)[4][4])l_s2.preact, l_s2.bias);
	apply_step_function<<<128, 128>>>(l_s2.preact, l_s2.output, l_s2.O);
	
	fprintf(stdout ,"Conv3 Forwarding\n");
	fp_preact_c3<<<128, 128>>>((float (*)[4][4])l_s2.output, l_c3.preact, (float (*)[16][4][4])l_c3.weight);
	fp_bias_c3<<<128, 128>>>(l_c3.preact, l_c3.bias);
	apply_step_function<<<128, 128>>>(l_c3.preact, l_c3.output, l_c3.O);
	
	fprintf(stdout ,"Full1 Forwarding\n");
	fp_preact_f1<<<128, 128>>>(l_c3.output, l_f1.preact, (float (*)[120])l_f1.weight);
	fp_bias_f1<<<128, 128>>>(l_f1.preact, l_f1.bias);
	apply_step_function<<<128, 128>>>(l_f1.preact, l_f1.output, l_f1.O);
	
	fprintf(stdout ,"Full2 Forwarding\n");
	fp_preact_f2<<<128, 128>>>(l_f2.output, l_f2.preact, (float (*)[84])l_f2.weight);
	fp_bias_f2<<<128, 128>>>(l_f2.preact, l_f2.bias);
	apply_step_function<<<128, 128>>>(l_f2.preact, l_f2.output, l_f2.O);

	fprintf(stdout ,"forward pass done!!\n");
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
	clock_t start, end;

	start = clock();
	fprintf(stdout ,"Full2 Backwarding\n");
	bp_weight_f2<<<128, 128>>>((float (*)[84])l_f2.d_weight, l_f2.d_preact, l_f1.output);
	bp_bias_f2<<<128, 128>>>(l_f2.bias, l_f2.d_preact);
	
	fprintf(stdout ,"Full1 Backwarding\n");
	bp_output_f1<<<128, 128>>>(l_f1.d_output, (float (*)[84])l_f2.weight, l_f2.d_preact);
	bp_preact_f1<<<128, 128>>>(l_f1.d_preact, l_f1.d_output, l_f1.preact);
	bp_weight_f1<<<128, 128>>>((float (*)[120])l_f1.d_weight, l_f1.d_preact, l_c3.output);
	bp_bias_f1<<<128, 128>>>(l_f1.bias, l_f1.d_preact);

	fprintf(stdout ,"Conv3 Backwarding\n");
	bp_output_c3<<<128, 128>>>(l_c3.d_output, (float (*)[120])l_f1.weight, l_f1.d_preact);
	bp_preact_c3<<<128, 128>>>(l_c3.d_preact, l_c3.d_output, l_c3.preact);
	bp_weight_c3<<<128, 128>>>((float (*)[16][4][4])l_c3.d_weight, l_c3.d_preact, (float (*)[4][4])l_s2.output);
	bp_bias_c3<<<128, 128>>>(l_c3.bias, l_c3.d_preact);

	fprintf(stdout ,"Pool2 Backwarding\n");
	bp_output_s2<<<128, 128>>>((float (*)[4][4])l_s2.d_output, (float (*)[16][4][4])l_c3.weight, l_c3.d_preact);
	bp_preact_s2<<<128, 128>>>((float (*)[4][4])l_s2.d_preact, (float (*)[4][4])l_s2.d_output, (float (*)[4][4])l_s2.preact);
	bp_weight_s2<<<128, 128>>>((float (*)[2][2])l_s2.d_weight, (float (*)[4][4])l_s2.d_preact, (float (*)[8][8])l_c2.output);
	bp_bias_s2<<<128, 128>>>(l_s2.bias, (float (*)[4][4])l_s2.d_preact);
	
	fprintf(stdout ,"Conv2 Backwarding\n");
	bp_output_c2<<<128, 128>>>((float (*)[8][8])l_c2.d_output, (float (*)[2][2])l_s2.weight, (float (*)[4][4])l_s2.d_preact);
	bp_preact_c2<<<128, 128>>>((float (*)[8][8])l_c2.d_preact, (float (*)[8][8])l_c2.d_output, (float (*)[8][8])l_c2.preact);
	bp_weight_c2<<<128, 128>>>((float (*)[6][5][5])l_c2.d_weight, (float (*)[8][8])l_c2.d_preact, (float (*)[12][12])l_s1.output);
	bp_bias_c2<<<128, 128>>>(l_c2.bias, (float (*)[8][8])l_c2.d_preact);

	fprintf(stdout ,"Pool1 Backwarding\n");
	bp_output_s1<<<128, 128>>>((float (*)[12][12])l_s1.d_output, (float (*)[6][5][5])l_c2.weight, (float (*)[8][8])l_c2.d_preact);
	bp_preact_s1<<<128, 128>>>((float (*)[12][12])l_s1.d_preact, (float (*)[12][12])l_s1.d_output, (float (*)[12][12])l_s1.preact);
	bp_weight_s1<<<128, 128>>>((float (*)[2][2])l_s1.d_weight, (float (*)[12][12])l_s1.d_preact, (float (*)[24][24])l_c1.output);
	bp_bias_s1<<<128, 128>>>(l_s1.bias, (float (*)[12][12])l_s1.d_preact);

	fprintf(stdout ,"Conv1 Backwarding\n");
	bp_output_c1<<<128, 128>>>((float (*)[24][24])l_c1.d_output, (float (*)[2][2])l_s1.weight, (float (*)[12][12])l_s1.d_preact);
	bp_preact_c1<<<128, 128>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
	bp_weight_c1<<<128, 128>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
	bp_bias_c1<<<128, 128>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);

	fprintf(stdout ,"Update Weight\n");
	apply_grad<<<128, 128>>>(l_f2.weight, l_f2.d_weight, l_f2.M * l_f2.N);
	apply_grad<<<128, 128>>>(l_f1.weight, l_f1.d_weight, l_f1.M * l_f1.N);
	apply_grad<<<128, 128>>>(l_c3.weight, l_c3.d_weight, l_c3.M * l_c3.N);
	apply_grad<<<128, 128>>>(l_s2.weight, l_s2.d_weight, l_s2.M * l_s2.N);
	apply_grad<<<128, 128>>>(l_c2.weight, l_c2.d_weight, l_c2.M * l_c2.N);
	apply_grad<<<128, 128>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad<<<128, 128>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);


	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}

static void learn()
{
	//static cublasHandle_t blas;
	//cublasCreate(&blas);

	float err;
	int iter = 1;
	
	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < 1; ++i) {
			float tmp_err;
			//fprintf(stdout ,"Before Forward_Pass\n %d\n", i);
			time_taken += forward_pass(train_set[i].data);
			//fprintf(stdout ,"After Forward_Pass\n");
			l_f2.bp_clear();
			l_f1.bp_clear();
			l_s2.bp_clear();
			l_s1.bp_clear();
			l_c3.bp_clear();
			l_c2.bp_clear();
			l_c1.bp_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(l_f2.d_preact, l_f2.output, train_set[i].label, 10);
			//cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
			tmp_err = 0.3;
			err += tmp_err;
			//fprintf(stdout ,"Before Backward_Pass\n");
			time_taken += back_pass();
			//fprintf(stdout ,"After Backward_Pass\n");
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);
		/*
		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}
		*/

	}
	
	fprintf(stdout, "\n Time - %lf\n", time_taken);
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_f2.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}
