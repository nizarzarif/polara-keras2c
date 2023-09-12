/**
k2c_convolution_layers.c
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c
 */


#include <math.h>
#include <stdio.h>
#include <string.h>
#include "k2c_include.h"
#include "fconv2d_tensor32.h"
#include "printf.h"
#define MAX_PAD 2





//fetch the filter value 


void NCHW_to_NHWC_tensor32_vec_16xC(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W) {

	uint64_t ldo = (W*C) << 2;			// Jump depth x column nb at each store
	uint64_t stride = W << 2;			// scalar values of each vectorn is stored C adress appart
	uint64_t ldi = C << 2;				// 
	int64_t block_size_o = 16; //16 input and 16 output at each iteration


  
for (int64_t r = 0; r < R; r += block_size_o) {

	asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(C));
	
	// Fetch 32 input vectors
	
	asm volatile("vle32.v v0,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v2,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v4,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v6,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v8,  (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v10, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v12, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v14, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v16, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v18, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v20, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v22, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v24, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v26, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v28, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	asm volatile("vle32.v v30, (%0); add %0, %0, %1" : "+&r"(i) : "r"(ldi));
	
	// Store each vector with the right stride 

	asm volatile("vsse32.v v0,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v2,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v4,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v6,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v8,  (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v10, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v12, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v14, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v16, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v18, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v20, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v22, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v24, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v26, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v28, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	asm volatile("vsse32.v v30, (%0), %1 ; add %0, %0, %2" : "+&r"(o), "+&r"(stride) : "r"(ldo));
	}
	

}

void NCHW_to_k2c_tensor32(k2c_tensor* dest_tensor, float* src_array, size_t N, size_t C, size_t H, size_t W) {
    // Note: Here, we assume that the destination tensor has already been allocated with the correct dimensions and sizes
    // shape[0] = H, shape[1] = W, shape[2] = C, shape[3] = N, shape[4] = 1

    for (size_t n = 0; n < N; n++) {       // Iterate over batch
        for (size_t h = 0; h < H; h++) {   // Iterate over height
            float *input = src_array + n * C * H * W + h * W * C; // NCHW format
            float *output = dest_tensor->array + n * H * W * C + h * W * C; // NHWC format
            
            NCHW_to_NHWC_tensor32_vec_16xC(output, input, H, W, C);
        }
    }
}





/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                            Description : Apply ReLu function to a tensor                                    //		
//                                                                                                             //				
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// *o : tensor output pointer C_in x H_in x W_in
// *i : input tensor pointer  C_in x H_in x W_in
// H_in  : number of input rows
// W_in  : number of input column
// C_in  : number of input channels


// Calculate 2 output matrix rows
void fReLu_tensor32(float *o, float *i, size_t H_in, size_t W_in, size_t C_in) {

int64_t const size = H_in * W_in * C_in;

float comp = 0;

asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(TILE_SIZE));

	for (int c = 0 ; c < size ; c += TILE_SIZE) // IF CONVOLUTION NEED TO BE TILED (C > TILE_SIZE)
	{
	
	  float *i_ = i + c;  // input pointer realtive to the tile (constant throughout the tile)
	  float *o_ = o + c;  // output pointer relative to the tile	
		
		
	  if(c > size - TILE_SIZE) 	// if we are at the right border of the input
				asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(size % TILE_SIZE));
	  
	  asm volatile("vle32.v v16,  (%0)" : "+&r"(i_));
	  
	  asm volatile("vfmax.vf v0,  v16,  %0" :: "f"(comp));
	  
	  asm volatile("vse32.v  v0,  (%0)" : "+&r"(o_));
	
	}

}

/**
 * 1D (temporal) Padding.
 *
 * :param output: tensor to store padded output data.
 * :param input: tensor to pad.
 * :param fill: value to fill in padded areas.
 * :param pad: array[2] of how many rows to pad. Order is {before dim 1, after dim 1}.
 */
void k2c_pad1d(k2c_tensor* output, const k2c_tensor* input, const float fill,
               const size_t * pad) {

    const size_t in_width = input->shape[1];
    const size_t pad_top = pad[0];

    // set output array to fill value
    if (fabs(fill) < 1e-6) {
        // fill is ~zero, use memset
        memset(output->array,0,output->numel*sizeof(output->array[0]));
    }
    else {
        for(size_t i=0; i<output->numel; ++i) {
            output->array[i] = fill;
        }
    }

    // memcpy the old array in the right place
    const size_t offset = pad_top*in_width;
    memcpy(&output->array[offset],&input->array[0],
           input->numel*sizeof(input->array[0]));
}


/**
 * Pad a 2D tensor with symmetric padding.
 *
 * @param output  Pointer to the output tensor.
 * @param input   Pointer to the input tensor.
 * @param fill    Value to fill the padded regions with.
 * @param pad     Array of padding values [top, bottom, left, right].
 *                Assumes symmetric padding, so only the top padding value is used.
 */
 /*
void k2c_pad2d(k2c_tensor* output, const k2c_tensor* input, const float fill,
    const size_t* pad) {
    // Calculate input tensor dimensions
    const size_t in_height = input->shape[0];
    const size_t in_width = input->shape[1];
    const size_t in_channels = input->shape[2];

    // Calculate padding values
    const size_t padding = pad[0];  // Assuming symmetric padding, taking the top padding value

    // Calculate the new dimensions after padding
    const size_t out_height = in_height + 2 * padding;
    const size_t out_width = in_width + 2 * padding;

    // set output array to fill value
    if (fabs(fill) < 1e-6) {
        // fill is ~zero, use memset
        memset(output->array, 0, output->numel * sizeof(output->array[0]));
    }
    else {
        for (size_t i = 0; i < output->numel; ++i) {
            output->array[i] = fill;
        }
    }

    // memcpy the old array in the middle with padding
    const size_t num = in_channels * in_width;
    const size_t padded_num = in_channels * out_width;  // Adjusted for padded width
    const size_t step = padded_num;

    // Calculate the offset for symmetric padding
    size_t offset = in_channels * padding * out_width + in_channels * padding;

    for (size_t i = 0; i < in_height; ++i) {
        memcpy(&output->array[offset],
            &input->array[i * num],
            num * sizeof(input->array[0]));
        offset += step;
    }

    // Check for NaN values in the output array
    int nan_count = containsNaN(output->array, output->numel);
    if (nan_count > 0) {
        printf("The output array contains %d NaN.\n", nan_count);
    }
}
*/
/**
 * 2D (spatial) Padding.
 *
 * :param output: tensor to store padded output data.
 * :param input: tensor to pad.
 * :param fill: value to fill in padded areas.
 * :param pad: array[4] of how many rows/cols to pad. Order is {before dim 1, after dim 1, before dim 2, after dim 2}.
 */

void k2c_pad2d(k2c_tensor* output, const k2c_tensor* input, const float fill,
               const size_t * pad) {

    const size_t in_height = input->shape[0];
    const size_t in_width = input->shape[1];
    const size_t in_channels = input->shape[2];
    const size_t pad_top = pad[0];
    const size_t pad_left = pad[2];
    const size_t pad_right = pad[3];
    int nan_count = 0;
  

    containsNaN(input->array, input->numel,"input->array");
    // set output array to fill value
    if (fabs(fill) < 1e-6) {
        // fill is ~zero, use memset
        memset(output->array,0,output->numel*sizeof(output->array[0]));
    }
    else {
        for(size_t i=0; i<output->numel; ++i) {
            output->array[i] = fill;
        }
    }
    // memcpy the old array in the middle
    size_t offset = in_channels*(pad_left+pad_right+in_width)*pad_top +
                    in_channels*pad_left;
    const size_t num = in_channels*in_width;
    const size_t step = num+in_channels*(pad_left+pad_right);
    for (size_t i=0; i<in_height; ++i) {
        memcpy(&output->array[offset],
               &input->array[i*num],
               num*sizeof(input->array[0]));
        offset += step;
    }
    containsNaN(output->array, output->numel,"output->array");

}


/**
 * 3D (spatial or spatio-temporal) Padding.
 *
 * :param output: tensor to store padded output data.
 * :param input: tensor to pad.
 * :param fill: value to fill in padded areas.
 * :param pad: array[6] of how many rows/cols to pad. Order is {before dim 1, after dim 1, before dim 2, after dim 2, before dim 3, after dim 3}.
 */
void k2c_pad3d(k2c_tensor* output, const k2c_tensor* input, const float fill,
               const size_t * pad) {

    const size_t dim1 = input->shape[0];
    const size_t dim2 = input->shape[1];
    const size_t dim3 = input->shape[2];
    const size_t outdim1 = dim1 + pad[0] + pad[1];
    const size_t outdim2 = dim2 + pad[2] + pad[3];
    const size_t outdim3 = dim3 + pad[4] + pad[5];
    const size_t in_channels = input->shape[3];

    // set output array to fill value
    if (fabs(fill) < 1e-6) {
        // fill is ~zero, use memset
        memset(output->array,0,output->numel*sizeof(output->array[0]));
    }
    else {
        for(size_t i=0; i<output->numel; ++i) {
            output->array[i] = fill;
        }
    }
    // memcpy the old array in the middle
    const size_t offset1 = in_channels*(outdim2*outdim3)*pad[0] + in_channels*outdim3*pad[2] + in_channels*pad[4];
    const size_t num = in_channels*dim3;
    const size_t outstep2 = num+in_channels*(pad[4]+pad[5]);
    const size_t outstep1 = outdim2*outdim3*in_channels;
    const size_t instep1 = dim2*dim3*in_channels;
    const size_t instep2 = dim3*in_channels;

    for (size_t i=0; i<dim1; ++i) {
        for (size_t j=0; j<dim2; ++j) {
            memcpy(&output->array[offset1+i*outstep1 + j*outstep2],
                   &input->array[i*instep1+j*instep2],
                   num*sizeof(input->array[0]));
        }
    }
}


/**
 * 1D (temporal) Convolution.
 * Assumes a "channels last" structure.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param kernel: kernel tensor.
 * :param bias: bias tensor.
 * :param stride: stride length of the convolution.
 * :param dilation: dilation rate to use for dilated convolution.
 * :param activation: activation function to apply to output.
 */
void k2c_conv1d(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
                const k2c_tensor* bias, const size_t stride, const size_t dilation,
                k2c_activationType *activation) {

    memset(output->array,0,output->numel*sizeof(output->array[0]));

    const size_t out_times = output->shape[0];
    const size_t out_channels = output->shape[1];
    const size_t in_channels = input->shape[1];

    for (size_t x0=0; x0 < out_times; ++x0) {
        for (size_t z=0; z < kernel->shape[0]; ++z) {
            for (size_t q=0; q < in_channels; ++q) {
                for (size_t k=0; k < out_channels; ++k) {
                    output->array[x0*out_channels + k] +=
                        kernel->array[z*(kernel->shape[2]*kernel->shape[1]) +
                                                                            q*(kernel->shape[2]) + k]*
                        input->array[(x0*stride + dilation*z)*in_channels + q];
                }
            }
        }
    }
    k2c_bias_add(output,bias);
    activation(output->array,output->numel);
}


/**
 * 2D (spatial) Convolution.
 * Assumes a "channels last" structure.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param kernel: kernel tensor.
 * :param bias: bias tensor.
 * :param stride: array[2] of stride length of the convolution. Order is {stride dim 1, stride dim 2}.
 * :param dilation: array[2] dilation rate to use for dilated convolution. Order is {dilation dim 1, dilation dim 2}.
 * :param activation: activation function to apply to output.
 */
void k2c_conv2d_fixed_point(k2c_tensor_int* output, const k2c_tensor_int* input, const k2c_tensor_int* kernel,
    const k2c_tensor_int* bias, const size_t* stride, const size_t* dilation,
    k2c_activationType_int* activation, size_t shift_factor, size_t scale_factor) {

    memset(output->array, 0, output->numel * sizeof(output->array[0]));

    const size_t out_rows = output->shape[0];
    const size_t out_cols = output->shape[1];
    const size_t out_channels = output->shape[2];
    const size_t in_channels = input->shape[2];
    //const size_t in_channels = 1;
    int temp1, temp2, temp3, counter;
    for (size_t x0 = 0; x0 < out_rows; ++x0) {
        for (size_t x1 = 0; x1 < out_cols; ++x1) {
            for (size_t z0 = 0; z0 < kernel->shape[0]; ++z0) {
                for (size_t z1 = 0; z1 < kernel->shape[1]; ++z1) {
                    for (size_t q = 0; q < in_channels; ++q) {
                        for (size_t k = 0; k < out_channels; ++k) {
                            temp1 = x0 * (output->shape[2] * output->shape[1])
                                + x1 * (output->shape[2]) + k;
                            temp2 = z0 * (kernel->shape[3] * kernel->shape[2] * kernel->shape[1])
                                + z1 * (kernel->shape[3] * kernel->shape[2])
                                + q * (kernel->shape[3]) + k;
                            temp3 = (x0 * stride[0]
                                + dilation[0] * z0) * (in_channels * input->shape[1])
                                + (x1 * stride[1] + dilation[1] * z1) * (in_channels)+q;
                            //printf("temp3 = %d, MUlresult = %d\n",temp3, multiplyFixedPoint(kernel->array[temp2], input->array[temp3], scale_factor, scale_factor));
                            int increment = multiplyFixedPoint(kernel->array[temp2], input->array[temp3], shift_factor, shift_factor);
                            output->array[temp1] = addFixedPoint(output->array[temp1], increment, shift_factor, shift_factor);
                            /*kernel->array[z0 * (kernel->shape[3] * kernel->shape[2] * kernel->shape[1])
                                          + z1*(kernel->shape[3]*kernel->shape[2])
                                          + q*(kernel->shape[3]) + k]*
                            input->array[(x0*stride[0]
                                          + dilation[0]*z0)*(in_channels *input->shape[1])
                                         + (x1*stride[1] + dilation[1]*z1)*(input->shape[2]) + q];*/

                        }
                    }
                }
            }
        }
    }
    
    k2c_bias_add_fixed_point(output, bias);
    activation(output->array, output->numel, shift_factor);
}


void transpose_kernel(k2c_tensor* kernel) {
    size_t H = kernel->shape[0];
    size_t W = kernel->shape[1];
    size_t Cin = kernel->shape[2];
    size_t Cout = kernel->shape[3];
    
    float buffer[K2C_MAX_NDIM * K2C_MAX_NDIM * K2C_MAX_NDIM * K2C_MAX_NDIM]; // assuming this doesn't exceed your system's static memory limit
    
    // Copy kernel to buffer
    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            for (size_t cin = 0; cin < Cin; ++cin) {
                for (size_t cout = 0; cout < Cout; ++cout) {
                    buffer[cout * Cin * H * W + cin * H * W + h * W + w] = kernel->array[h * W * Cin * Cout + w * Cin * Cout + cin * Cout + cout];
                }
            }
        }
    }
    
    // Copy buffer back to kernel
    for (size_t i = 0; i < H*W*Cin*Cout; ++i) {
        kernel->array[i] = buffer[i];
    }
}

float static_buffer_for_input[K2C_MAX_NDIM * K2C_MAX_NDIM * K2C_MAX_NDIM]; // adjust as per the maximum tensor size you're working with

void in_place_nhwc_to_nchw_static(k2c_tensor* tensor) {
    size_t h = tensor->shape[0];
    size_t w = tensor->shape[1];
    size_t c = tensor->shape[2];
    
    for (size_t channel = 0; channel < c; channel++) {
        for (size_t height = 0; height < h; height++) {
            for (size_t width = 0; width < w; width++) {
                static_buffer_for_input[channel * h * w + height * w + width] = tensor->array[height * w * c + width * c + channel];
            }
        }
    }
    
    for (size_t i = 0; i < h * w * c; ++i) {
        tensor->array[i] = static_buffer_for_input[i];
    }
}

void in_place_nchw_to_nhwc_static(k2c_tensor* tensor) {
    size_t h = tensor->shape[1];  // Assuming NCHW format
    size_t w = tensor->shape[2];
    size_t c = tensor->shape[0];
    
    for (size_t channel = 0; channel < c; channel++) {
        for (size_t height = 0; height < h; height++) {
            for (size_t width = 0; width < w; width++) {
                static_buffer_for_input[height * w * c + width * c + channel] = tensor->array[channel * h * w + height * w + width];
            }
        }
    }
    
    for (size_t i = 0; i < h * w * c; ++i) {
        tensor->array[i] = static_buffer_for_input[i];
    }
}

void transposeOutput(k2c_tensor* tensor) {
    int out_channels = tensor->shape[3]; // Cout
    int in_channels = tensor->shape[2];  // Cin
    int out_rows = tensor->shape[0];     // Height
    int out_cols = tensor->shape[1];     // Width

    // Stack-allocated temporary buffer
    float tempBuffer[out_channels * in_channels * out_rows * out_cols];

    // Perform transposition into tempBuffer
    for (int k_out = 0; k_out < out_channels; k_out++) {
        for (int k_in = 0; k_in < in_channels; k_in++) {
            for (int r = 0; r < out_rows; r++) {
                for (int c = 0; c < out_cols; c++) {
                    int curr_index = k_out*(in_channels*out_rows*out_cols) 
                                    + k_in*(out_rows*out_cols) 
                                    + r*out_cols + c;

                    int target_index = r*(out_cols*in_channels*out_channels) 
                                      + c*(in_channels*out_channels) 
                                      + k_in*out_channels + k_out;

                    tempBuffer[target_index] = tensor->array[curr_index];
                }
            }
        }
    }

    // Copy the transposed data from tempBuffer back into tensor->array
    for (int i = 0; i < out_channels * in_channels * out_rows * out_cols; i++) {
        tensor->array[i] = tempBuffer[i];
    }
}

void transposeOutput_vector(k2c_tensor* tensor) {
    int N = tensor->shape[3];            // Batch
    int C = tensor->shape[2];            // Channels
    int H = tensor->shape[0];            // Height
    int W = tensor->shape[1];            // Width

    // Assuming float is the underlying data type for the tensor, which matches with the provided NCHW_to_NHWC_tensor32_vec_16xC function
    float* src_data = tensor->array;

    // Loop through batches and height, then convert from NCHW to NHWC in-place
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            float* input_data = src_data + n * C * H * W + h * W * C;  // Pointer to input data in NCHW format
            float* output_data = input_data;                           // Since the operation is in-place, output is same as input
            
            NCHW_to_NHWC_tensor32_vec_16xC(output_data, input_data, H, W, C);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                           Description : Functions for cross-correlation between                             //		
//                                                                                                             //
//                      1 x Cin x Hin x Win  * Cout x Cin x F x F   =    Cout x Hout x Wout                    //			
//                          input (32b)            kernels (32b)             output (32b)                      //	
//																																					//
//																																					//				
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// *o    : tensor convolution output pointer 
// *i    : input tensor pointer
// *f    : kernel/filter tensor pointer
// R   : number of input cows
// C   : number of input column
// W   : number of input channels 
// F     : size of the kernel/filter 
// K  : number of kernel/filter corresponding to the number of output channels
void fconv2d_tensor32_naive(float *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K) {
//treat pointers as 3D arrays
float (*i_)[R+F-1][C+F-1] = (float (*)[R+F-1][C+F-1])i;
float (*f_)[W][F][F] = (float (*)[W][F][F])f;
float (*o_)[R][C] = (float (*)[R][C])o;
 
for(int k = 0 ; k < K ; k++) 
	for(int ch = 0 ; ch < W ; ch++)
		for(int r = 0 ; r < R ; r++)
			for(int c = 0 ; c < C ; c++)
				for(int fh = 0 ; fh < F ; fh++)
					for(int fw = 0 ; fw < F ; fw++) {
						o_[k][r][c] += i_[ch][r+fh][c+fw]*f_[k][ch][fh][fw];
					}
}
/**
 * 2D (spatial) Convolution.
 * Assumes a "channels last" structure.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param kernel: kernel tensor.
 * :param bias: bias tensor.
 * :param stride: array[2] of stride length of the convolution. Order is {stride dim 1, stride dim 2}.
 * :param dilation: array[2] dilation rate to use for dilated convolution. Order is {dilation dim 1, dilation dim 2}.
 * :param activation: activation function to apply to output.
 */
void k2c_conv2d(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
                const k2c_tensor* bias, const size_t * stride, const size_t * dilation,
                k2c_activationType *activation) {

    memset(output->array,0,output->numel*sizeof(output->array[0]));
    #ifndef VLEN
    const size_t out_rows = output->shape[0];
    const size_t out_cols = output->shape[1];
    const size_t out_channels = output->shape[2];
    const size_t in_channels = input->shape[2];

    printf("conv2d on scalar core\n");
    
    for (size_t x0=0; x0 < out_rows; ++x0) {
        for (size_t x1=0; x1 < out_cols; ++x1) {
            for (size_t z0=0; z0 < kernel->shape[0]; ++z0) {
                for (size_t z1=0; z1 < kernel->shape[1]; ++z1) {
                    for (size_t q=0; q < in_channels; ++q) {
                        for (size_t k=0; k < out_channels; ++k) {
                            output->array[x0*(output->shape[2]*output->shape[1])
                                          + x1*(output->shape[2]) + k] +=
                                              kernel->array[z0*(kernel->shape[3]*kernel->shape[2]*kernel->shape[1])
                                                            + z1*(kernel->shape[3]*kernel->shape[2])
                                                            + q*(kernel->shape[3]) + k]*
                                              input->array[(x0*stride[0]
                                                            + dilation[0]*z0)*(input->shape[2]*input->shape[1])
                                                           + (x1*stride[1] + dilation[1]*z1)*(input->shape[2]) + q];
                        }
                    }
                }
            }
        }
    }
    //print_tensor_values(output);
    k2c_bias_add(output,bias);
    activation(output->array,output->numel);
    #else
    printf("conv2d on vector core\n");


// Assuming this function only handles 'valid' padding for simplicity
    //size_t out_rows = (input->shape[0] -kernel->shape[0] + 1);
    //size_t out_cols = (input->shape[1] - kernel->shape[0]+ 1);
    
    //print_tensor_values(input);
    // Convert input from NHWC to NCHW in-place
    in_place_nhwc_to_nchw_static(input);

    // print_tensor_values(input);
    //print_tensor_values(kernel);
    // Transpose kernel
    transpose_kernel(kernel);
    //print_tensor_values(kernel);
    // Call the convolution function
    //fconv2d_tensor32_naive(output->array, input->array, kernel->array, out_rows, out_cols, input->shape[2], kernel->shape[0], kernel->shape[3]);
    fconv2d_tensor32(output->array, input->array, kernel->array, input->shape[0], input->shape[1], input->shape[2], kernel->shape[0], kernel->shape[3]);

    //print_tensor_values(output);
    // Convert output from NCHW to NHWC in-place
    transposeOutput(output);
    //transposeOutput_vector(output);
    //NCHW_to_k2c_tensor32(output, output->array,output->shape[2], output->shape[3], output->shape[0], output->shape[1]);
    //NCHW_to_NHWC_tensor32(output);
 
    //print_tensor_values(output);
    k2c_bias_add(output,bias);
    fReLu_tensor32(output->array, output->array, output->shape[0], output->shape[1], output->shape[2]);
    #endif
     
}


/**
 * 3D (spatial or spatio-temporal) Convolution.
 * Assumes a "channels last" structure.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param kernel: kernel tensor.
 * :param bias: bias tensor.
 * :param stride: array[3] of stride length of the convolution. Order is {stride dim 1, stride dim 2, stride dim 3}.
 * :param dilation: array[3] dilation rate to use for dilated convolution. Order is {dilation dim 1, dilation dim 2, dilation dim 3}.
 * :param activation: activation function to apply to output.
 */
void k2c_conv3d(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
                const k2c_tensor* bias, const size_t * stride, const size_t * dilation,
                k2c_activationType *activation) {

    memset(output->array,0,output->numel*sizeof(output->array[0]));
    const size_t dim1 = output->shape[0];
    const size_t dim2 = output->shape[1];
    const size_t dim3 = output->shape[2];
    const size_t out_channels = output->shape[3];
    const size_t in_channels = input->shape[3];

    for (size_t x0=0; x0 < dim1; ++x0) {
        for (size_t x1=0; x1 < dim2; ++x1) {
            for (size_t x2=0; x2<dim3; ++x2) {
                for (size_t z0=0; z0 < kernel->shape[0]; ++z0) {
                    for (size_t z1=0; z1 < kernel->shape[1]; ++z1) {
                        for (size_t z2=0; z2 < kernel->shape[2]; ++z2) {
                            for (size_t q=0; q < in_channels; ++q) {
                                for (size_t k=0; k < out_channels; ++k) {
                                    output->array[x0*(output->shape[3]*output->shape[2]
                                                      *output->shape[1])
                                                  + x1*(output->shape[3]*output->shape[2])
                                                  + x2*(output->shape[3]) + k] +=
                                                      kernel->array[z0*(kernel->shape[4]*kernel->shape[3]
                                                                        *kernel->shape[2]*kernel->shape[1])
                                                                    + z1*(kernel->shape[4]*kernel->shape[3]
                                                                          *kernel->shape[2])
                                                                    + z2*(kernel->shape[4]*kernel->shape[3])
                                                                    + q*(kernel->shape[4]) + k]
                                                      *input->array[(x0*stride[0] + dilation[0]*z0)
                                                                    *(input->shape[3]*input->shape[2]*input->shape[1])
                                                                    + (x1*stride[1] + dilation[1]*z1)
                                                                    *(input->shape[3]*input->shape[2])
                                                                    + (x2*stride[2] + dilation[2]*z2)
                                                                    *(input->shape[3]) + q];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    k2c_bias_add(output,bias);
    activation(output->array,output->numel);
}


/**
 * 1D (temporal) Cropping.
 *
 * :param output: tensor to store cropped output data.
 * :param input: tensor to crop.
 * :param pad: array[2] of how many rows to crop. Order is {before dim 1, after dim 1}.
 */
void k2c_crop1d(k2c_tensor* output, const k2c_tensor* input, const size_t * crop) {

    const size_t offset = crop[0]*input->shape[1];
    memcpy(&output->array[0],&input->array[offset],
           output->numel*sizeof(output->array[0]));
}


/**
 * 2D (spatial) Cropping.
 *
 * :param output: tensor to store cropped output data.
 * :param input: tensor to crop.
 * :param pad: array[4] of how many rows/cols to crop. Order is {before dim 1, after dim 1, before dim 2, after dim 2}.
 */
void k2c_crop2d(k2c_tensor* output, const k2c_tensor* input, const size_t * crop) {

    const size_t out_height = output->shape[0];
    const size_t in_width = input->shape[1];
    const size_t in_channels = input->shape[2];
    const size_t crop_top = crop[0];
    const size_t crop_left = crop[2];
    const size_t crop_right = crop[3];

    size_t offset = in_channels*in_width*crop_top + in_channels*crop_left;
    const size_t num = in_channels*(in_width-crop_left-crop_right);
    for (size_t i=0; i<out_height; ++i) {
        memcpy(&output->array[i*num],&input->array[offset],num*sizeof(input->array[0]));
        offset += in_width*in_channels;
    }
}


/**
 * 3D (spatial or spatio-temporal) Cropping.
 *
 * :param output: tensor to store cropped output data.
 * :param input: tensor to crop.
 * :param pad: array[6] of how many rows/cols to crop. Order is {before dim 1, after dim 1, before dim 2, after dim 2, before dim 3, after dim 3}.
 */
void k2c_crop3d(k2c_tensor* output, const k2c_tensor* input, const size_t * crop) {

    const size_t dim1 = input->shape[0];
    const size_t dim2 = input->shape[1];
    const size_t dim3 = input->shape[2];
    const size_t outdim1 = dim1 - crop[0] - crop[1];
    const size_t outdim2 = dim2 - crop[2] - crop[3];
    const size_t outdim3 = dim3 - crop[4] - crop[5];
    const size_t in_channels = input->shape[3];

    const size_t offset1 = in_channels*(dim2*dim3)*crop[0] +
                           in_channels*dim3*crop[2] + in_channels*crop[4];
    const size_t num = in_channels*outdim3;
    const size_t instep2 = num+in_channels*(crop[4]+crop[5]);
    const size_t instep1 = dim2*dim3*in_channels;
    const size_t outstep1 = outdim2*outdim3*in_channels;
    const size_t outstep2 = outdim3*in_channels;

    for (size_t i=0; i<outdim1; ++i) {
        for (size_t j=0; j<outdim2; ++j) {
            memcpy(&output->array[i*outstep1 + j*outstep2],
                   &input->array[offset1+i*instep1+j*instep2],
                   num*sizeof(input->array[0]));
        }
    }
}


/**
 * 1D (temporal) Upsampling.
 * Repeats each temporal step size times along the time axis.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param size: Upsampling factor.
 */
void k2c_upsampling1d(k2c_tensor* output, const k2c_tensor* input, const size_t size) {

    const size_t in_height = input->shape[0];
    const size_t in_width = input->shape[1];

    for (size_t i=0; i<in_height; ++i) {
        for (size_t j=0; j<size; ++j) {
            for (size_t k=0; k<in_width; ++k) {
                output->array[(size*i+j)*in_width + k] = input->array[i*in_width+k];
            }
        }
    }
}


/**
 * 2D (spatial) Upsampling.
 * Repeats the rows and columns of the data by size[0] and size[1] respectively.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param size: array[2] of upsampling factors. Order is {upsampling dim 1, upsampling dim 2}.
 */
void k2c_upsampling2d(k2c_tensor* output, const k2c_tensor* input, const size_t * size) {

    const size_t out_height = output->shape[0];
    const size_t out_width = output->shape[1];
    const size_t channels = input->shape[2];

    for (size_t i=0; i<out_height; ++i) {
        for (size_t j=0; j<out_width; ++j) {
            const size_t insub[K2C_MAX_NDIM] = {i/size[0],j/size[1],0};
            const size_t outsub[K2C_MAX_NDIM] = {i,j,0};
            memcpy(&output->array[k2c_sub2idx(outsub,output->shape,output->ndim)],
                   &input->array[k2c_sub2idx(insub,input->shape,input->ndim)],
                   channels*sizeof(input->array[0]));
        }
    }
}


/**
 * 2D (spatial) Upsampling.
 * Repeats the 1st, 2nd and 3rd dimensions of the data by size[0], size[1] and size[2] respectively.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param size: array[3] of upsampling factors. Order is {upsampling dim 1, upsampling dim 2, upsampling dim 3}.
 */
void k2c_upsampling3d(k2c_tensor* output, const k2c_tensor* input, const size_t * size) {

    const size_t dim1 = output->shape[0];
    const size_t dim2 = output->shape[1];
    const size_t dim3 = output->shape[2];
    const size_t channels = input->shape[3];

    for (size_t i=0; i<dim1; ++i) {
        for (size_t j=0; j<dim2; ++j) {
            for (size_t k=0; k<dim3; ++k) {
                const size_t insub[K2C_MAX_NDIM] = {i/size[0],j/size[1],k/size[2],0};
                const size_t outsub[K2C_MAX_NDIM] = {i,j,k,0};
                memcpy(&output->array[k2c_sub2idx(outsub,output->shape,output->ndim)],
                       &input->array[k2c_sub2idx(insub,input->shape,input->ndim)],
                       channels*sizeof(input->array[0]));
            }
        }
    }
}

/**
 * Performs separable convolution operation on a 2D input tensor.
 *
 * @param output          Output tensor to store the result of the convolution.
 * @param input           Input tensor for the convolution operation.
 * @param depthwise_kernel    Depthwise convolution kernel tensor.
 * @param pointwise_kernel    Pointwise convolution kernel tensor.
 * @param bias            Bias tensor to be added to the output.
 * @param stride          Array specifying the stride values for both spatial dimensions.
 * @param dilation        Array specifying the dilation values for both spatial dimensions.
 * @param activation      Activation function pointer to apply to the output tensor.
 */
void k2c_separable_conv2d(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* depthwise_kernel,
    const k2c_tensor* pointwise_kernel, const k2c_tensor* bias,
    const size_t* stride, const size_t* dilation, k2c_activationType* activation) {

    // Initialize the output tensor with zeros
    memset(output->array, 0, output->numel * sizeof(output->array[0]));

    // Extract relevant dimensions for convenience
    const size_t out_rows = output->shape[0];
    const size_t out_cols = output->shape[1];
    const size_t out_channels = output->shape[2];
    const size_t in_channels = input->shape[2];

    // Perform separable convolution
    for (size_t x0 = 0; x0 < out_rows; ++x0) {
        for (size_t x1 = 0; x1 < out_cols; ++x1) {
            for (size_t z0 = 0; z0 < depthwise_kernel->shape[0]; ++z0) {
                for (size_t z1 = 0; z1 < depthwise_kernel->shape[1]; ++z1) {
                    for (size_t q = 0; q < in_channels; ++q) {
                        // Depthwise Convolution
                        const float depthwise_val = depthwise_kernel->array[z0 * (depthwise_kernel->shape[3] * depthwise_kernel->shape[2] * depthwise_kernel->shape[1])
                            + z1 * (depthwise_kernel->shape[3] * depthwise_kernel->shape[2])
                            + q * (depthwise_kernel->shape[3]) + 0] *
                            input->array[(x0 * stride[0] + dilation[0] * z0) * (input->shape[2] * input->shape[1])
                            + (x1 * stride[1] + dilation[1] * z1) * (input->shape[2]) + q];

                        for (size_t k = 0; k < out_channels; ++k) {
                            // Pointwise Convolution
                            output->array[x0 * (output->shape[2] * output->shape[1]) + x1 * (output->shape[2]) + k] +=
                                depthwise_val * pointwise_kernel->array[q * (pointwise_kernel->shape[1]) + k];
                        }
                    }
                }
            }
        }
    }

    // Add bias and apply activation function
    k2c_bias_add(output, bias);
    activation(output->array, output->numel);
}
