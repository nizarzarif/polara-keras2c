/**
k2c_core_layers.c
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "k2c_include.h"




 /**
  * Applies a dense (fully connected) layer to the input tensor using fixed-point arithmetic.
  *
  * @param output         Pointer to the output tensor, which stores the result of the dense layer
  * @param input          Pointer to the input tensor
  * @param kernel         Pointer to the kernel tensor (weights) of the dense layer
  * @param bias           Pointer to the bias tensor of the dense layer
  * @param activation     Function pointer to the activation function to be applied after the dense layer
  * @param fwork          Pointer to the temporary workspace for intermediate computations
  * @param shift_factor   The number of bits to shift the fixed-point values during multiplication
  * @param scale_factor   The scaling factor used to convert fixed-point result to floating point
  */
void k2c_dense_fixed_point(k2c_tensor_int* output, k2c_tensor_int* input, const k2c_tensor_int* kernel,
    const k2c_tensor_int* bias, k2c_activationType* activation, float* fwork,
    size_t shift_factor, size_t scale_factor) {
    printf("dense_input_input nb %d \n", input->ndim);

    if (input->ndim <= 2) {
        size_t outrows;

        if (input->ndim > 1) {
            outrows = input->shape[0];
        }
        else {
            outrows = 1;
        }

        const size_t outcols = kernel->shape[1];
        const size_t innerdim = kernel->shape[0];
        const size_t outsize = outrows * outcols;

        // Perform affine matrix multiplication between input and kernel, with bias
        k2c_affine_matmul(output->array, input->array, kernel->array, bias->array,
            outrows, outcols, innerdim, shift_factor, scale_factor);

        // Apply the activation function to the output tensor
        activation(output->array, outsize);
    }
    else {
        const size_t axesA[1] = { input->ndim - 1 };
        const size_t axesB[1] = { 0 };
        const size_t naxes = 1;
        const int normalize = 0;

        // Perform dot product between input and kernel, along specified axes
        k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork);

        // Add bias to the output tensor
        k2c_bias_add(output, bias);

        // Apply the activation function to the output tensor
        activation(output->array, output->numel);
    }
}


/**
 * Dense (fully connected) Layer.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param kernel: kernel tensor.
 * :param bias: bias tensor.
 * :param activation: activation function to apply to output.
 * :param fwork: array of working space, size(fwork) = size(input) + size(kernel)
 */
void k2c_dense(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
               const k2c_tensor* bias, k2c_activationType *activation, float * fwork) {

    if (input->ndim <=2) {
        size_t outrows;

        if (input->ndim>1) {
            outrows = input->shape[0];
        }
        else {
            outrows = 1;
        }
        const size_t outcols = kernel->shape[1];
        const size_t innerdim = kernel->shape[0];
        const size_t outsize = outrows*outcols;
        k2c_affine_matmul(output->array,input->array,kernel->array,bias->array,
                          outrows,outcols,innerdim);
        activation(output->array,outsize);
    }
    else {
        const size_t axesA[1] = {input->ndim-1};
        const size_t axesB[1] = {0};
        const size_t naxes = 1;
        const int normalize = 0;

        k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork);
        k2c_bias_add(output, bias);
        activation(output->array, output->numel);
    }
}


/**
 * Flatten Layer.
 * flattens inputs to ndim=1
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param kernel: kernel tensor.
 */
void k2c_flatten(k2c_tensor *output, const k2c_tensor* input) {

    memcpy(output->array, input->array, input->numel*sizeof(input->array[0]));
    for (size_t i=0; i<input->ndim; ++i) {
        output->shape[i] = 1;
    }
    output->shape[0] = input->numel;
    output->numel = input->numel;
    output->ndim = 1;
}

/**
 * Reshape Layer.
 * reshapes input to arbitrary output shape, while preserving total number of elements.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param newshp: array[newndim] of the desired new shape.
 * :param newndim: number of dimensions after reshaping.
 */
void k2c_reshape(k2c_tensor *output, const k2c_tensor* input, const size_t * newshp,
                 const size_t newndim) {

    memcpy(output->array, input->array, input->numel*sizeof(input->array[0]));
    for (size_t i=0; i<newndim; ++i) {
        output->shape[i] = newshp[i];
    }
    output->ndim = newndim;
    output->numel = input->numel;
}


/**
 * Permute Layer.
 * permutes the dimensions of the input according to a given pattern.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param permute: array[ndim] Permutation pattern. Indexing starts at 0. For instance, (1, 0) permutes the first and second dimension of the input.
 */
void k2c_permute_dims(k2c_tensor* output, const k2c_tensor* input,
                      const size_t * permute) {

    size_t Asub[K2C_MAX_NDIM];
    size_t Bsub[K2C_MAX_NDIM];
    size_t newshp[K2C_MAX_NDIM];
    size_t oldshp[K2C_MAX_NDIM];
    const size_t ndim = input->ndim;
    size_t bidx=0;
    for (size_t i=0; i<ndim; ++i) {
        oldshp[i] = input->shape[i];
    }
    for (size_t i=0; i<ndim; ++i) {
        newshp[i] = oldshp[permute[i]];
    }

    for (size_t i=0; i<input->numel; ++i) {
        k2c_idx2sub(i,Asub,oldshp,ndim);
        for (size_t j=0; j<ndim; ++j) {
            Bsub[j] = Asub[permute[j]];
        }
        bidx = k2c_sub2idx(Bsub,newshp,ndim);
        output->array[bidx] = input->array[i];
    }
}


/**
 * Repeat Vector Layer.
 * repeats the input n times.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param n: repetition factor.
 */
void k2c_repeat_vector(k2c_tensor* output, const k2c_tensor* input, const size_t n) {

    const size_t in_width = input->shape[0];
    for (size_t i=0; i<n; ++i) {
        for(size_t j=0; j<in_width; ++j) {
            output->array[i*in_width + j] = input->array[j];
        }
    }
}
