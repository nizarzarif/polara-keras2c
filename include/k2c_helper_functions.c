/**
k2c_helper_functions.c
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "k2c_include.h"

 /**
  * Finds the index of the maximum element in a float array.
  *
  * @param arr The input float array.
  * @param size The size of the array.
  * @return The index of the maximum element.
  */
int argmax(const float arr[], int size) {
    int maxIndex = 0;
    float maxValue = arr[0];

    // Iterate over the array to find the maximum element
    for (int i = 1; i < size; i++) {
        // Update the maximum value and its index if a larger value is found
        if (arr[i] > maxValue) {
            maxValue = arr[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

 /**
  * Compare two float arrays and calculate the percentage of error.
  *
  * @param array1 The first float array.
  * @param array2 The second float array.
  * @param size The size of the arrays.
  * @return The average percentage of error between the two arrays.
  */
double calculate_percentage_error(const float* array1, const float* array2, int size) {
    double error_sum = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = fabs(array1[i] - array2[i]);
        double percent_error = (diff / fabs(array1[i])) * 100.0;
        error_sum += percent_error;

        //printf("Difference at index %d: %f\n", i, diff);
    }
    return error_sum / size;
}
 
 /**
  * Prints the values of a tensor of up to 5 dimensions.
  *
  * @param tensor The tensor to print.
  */
void printTensor( k2c_tensor tensor) {
    size_t dim[K2C_MAX_NDIM];
    size_t stride[K2C_MAX_NDIM];

    // Initialize dimensions and strides
    for (size_t i = 0; i < tensor.ndim; i++) {
        dim[i] = tensor.shape[i];
        stride[i] = (i == 0) ? 1 : stride[i - 1] * dim[i - 1];
    }

    // Calculate total size and number of elements per sub-tensor
    size_t totalSize = tensor.numel;
    size_t subTensorSize = tensor.ndim > 1 ? tensor.numel / tensor.shape[0] : tensor.numel;

    // Print values recursively
    printTensorRecursive(tensor.array, dim, stride, totalSize, subTensorSize, tensor.ndim, 0);
}

void printTensorRecursive(float* array, size_t* dim, size_t* stride, size_t totalSize,
    size_t subTensorSize, size_t ndim, size_t currDim) {
    for (size_t i = 0; i < dim[currDim]; i++) {
        size_t offset = i * stride[currDim];

        if (currDim < ndim - 1) {
            printTensorRecursive(array + offset, dim, stride, totalSize, subTensorSize,
                ndim, currDim + 1);
        }
        else {
            for (size_t j = 0; j < subTensorSize; j++) {
                printf("%f ", array[offset + j]);
            }
            printf("\n");
        }
    }
}

/**
* Checks if the given float array contains any NaN (Not a Number) values.
*
* @param arr The float array to be checked.
* @param size The size of the array.
* @param array_name The name of the array to be displayed in the output message.
* @return The number of NaN values found in the array.
*/
int containsNaN_int(int * arr, int size, char* array_name) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (isnan((float)arr[i])) {
            count++;  // Array contains nan
        }
        else {
            ;
        }
    }
    if (count > 0) {
        printf("%s array contains %d NaN\n", array_name, count);
    }
    return count;  // return number of NaN
}
 /**
  * Checks if the given float array contains any NaN (Not a Number) values.
  *
  * @param arr The float array to be checked.
  * @param size The size of the array.
  * @param array_name The name of the array to be displayed in the output message.
  * @return The number of NaN values found in the array.
  */
int containsNaN(float * arr, int size, char * array_name) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (isnan(arr[i])) {
            count++;  // Array contains nan
        }
        else {
            ;
        }
    }
    if (count > 0) {
        printf("%s array contains %d NaN\n",array_name,count);
    }
    return count;  // return number of NaN
}

 /**
  * Measure the difference and percentage of error between fixed-point and floating-point convolution outputs.
  *
  * @param fp_conv2d_1_output Pointer to the array of fixed-point convolution outputs.
  * @param conv2d_1_output Pointer to the array of floating-point convolution outputs.
  * @param output_size The size of the output arrays.
  * @param shift_factor The shift factor used for converting floating-point values to fixed-point representation.
  */
void measure_conv2d_outputs(k2c_tensor_int* fp_tensor, k2c_tensor * float_tensor, int shift_factor) {
    // Measure the difference and percentage of error
    double diff_sum = 0.0;
    double percent_error_sum = 0.0;
    int output_size = fp_tensor->numel;
    float sum = 0;
    //containsNaN(fp_tensor->array, fp_tensor->numel,"fp_tensor->array");
    containsNaN(float_tensor->array, float_tensor->numel,"float_tensor->array");
    for (int i = 0; i < output_size; i++) {
        int fp_value = fp_tensor->array[i];
        float float_value = float_tensor->array[i] * pow(2.0, shift_factor);  // Convert float value to fixed-point

        double diff = fabs(fp_value - float_value);
        sum += float_value;

        diff_sum += diff;
        
    }

    double avg_diff = diff_sum / output_size;
    double avg_percent_error = percent_error_sum / output_size;

    printf("Average difference: %f\n", avg_diff);
    printf("Average percentage of error: %f%%\n", 100.0*diff_sum/sum);
}

 /**
   * Convert an array of floating-point values to Q16.16 fixed-point format.
   *
   * @param x_float The array of floating-point values to convert.
   * @param x_fixed The array of Q16.16 fixed-point values to store the converted values in.
   * @param size The number of elements in the input and output arrays.
   */
void float_array_to_fixed(float* x_float, int* x_fixed, size_t size) {
    for (int i = 0; i < size; i++) {
        x_fixed[i] = (int)(x_float[i] * 65536.0f + 0.5f);
        if (isnan(x_fixed[i])) {
            printf("error at i = %d\n",i);
        };
    }
}

/**
 * Convert an array of Q16.16 fixed-point values to floating-point format.
 *
 * @param x_fixed The array of Q16.16 fixed-point values to convert.
 * @param x_float The array of floating-point values to store the converted values in.
 * @param size The number of elements in the input and output arrays.
 */
void fixed_array_to_float(int* x_fixed, float* x_float, size_t size) {
    for (int i = 0; i < size; i++) {
        x_float[i] = ((float)x_fixed[i]) / 65536.0f;
    }
}

/**
 * Convert a k2c_tensor of floating-point values to Q16.16 fixed-point format.
 *
 * @param x_float The k2c_tensor of floating-point values to convert.
 * @param x_fixed The k2c_tensor of Q16.16 fixed-point values to store the converted values in.
 * @param size The number of elements in the input and output tensors.
 */
void float_tensor_to_fixed(k2c_tensor* x_float, k2c_tensor_int* x_fixed, size_t size) {
    for (int i = 0; i < size; i++) {
        x_fixed->array[i] = (int)(x_float->array[i] * 65536.0f + 0.5f);
    }
}

/**
 * Convert a k2c_tensor of Q16.16 fixed-point values to floating-point format.
 *
 * @param x_fixed The k2c_tensor of Q16.16 fixed-point values to convert.
 * @param x_float The k2c_tensor of floating-point values to store the converted values in.
 * @param size The number of elements in the input and output tensors.
 */
void fixed_tensor_to_float(k2c_tensor_int* x_fixed, k2c_tensor* x_float, size_t size) {
    for (int i = 0; i < size; i++) {
        x_float->array[i] = ((float)x_fixed->array[i]) / 65536.0f;
    }
}


/**
 * Adds two fixed-point numbers in Qm.n format and returns the result in the same format.
 *
 * @param a The first fixed-point number to add.
 * @param b The second fixed-point number to add.
 * @param m The number of bits in the integer part of the Qm.n format.
 * @param n The number of bits in the fractional part of the Qm.n format.
 * @return The sum of `a` and `b` in Qm.n format.
 */
int32_t addFixedPoint(int32_t a, int32_t b, int m, int n) {
    int q = n;  // The number of bits in the fractional part of the Qm.n format.
    int qFactor = (1 << q);  // Factor to convert between fixed-point and floating-point.
    int64_t result = ((int64_t)a << q) + ((int64_t)b << q);  // Shift the Qm.n numbers to align the decimal points and add them together.
    static int counter = 0;
    // Check if the result overflows the Qm.n format and clip it to the maximum or minimum value if necessary.
    int64_t maxVal = (1 << (m + n - 1)) - 1;
    int64_t minVal = -1 * maxVal - 1;
    if (result > maxVal * qFactor) {
        result = maxVal * qFactor;

    }
    else if (result < minVal * qFactor) {
        result = minVal * qFactor;
        counter++;
        printf("%d,\n", counter);
        // Handle underflow here
    }

    // Truncate the result to a signed 32-bit integer, shift it back by q bits, and return it.
    return (int32_t)(result >> q);
}



/**
 * Multiplies two Qm.n fixed-point numbers and returns the result as a signed 32-bit integer.
 * If the result overflows the Qm.n format, it is clipped to the maximum or minimum value.
 * @param a The first Qm.n fixed-point number to multiply.
 * @param b The second Qm.n fixed-point number to multiply.
 * @param m The number of bits in the integer part of the Qm.n format.
 * @param n The number of bits in the fractional part of the Qm.n format.
 * @return The result of the multiplication, truncated to a signed 32-bit integer.
 */
int32_t multiplyFixedPoint(int32_t a, int32_t b, int m, int n) {
    int q = n;  // The number of bits in the fractional part of the Qm.n format.
    int qFactor = (1 << q);  // Factor to convert between fixed-point and floating-point.
    int64_t result = ((int64_t)a * (int64_t)b);  // Multiply the two Qm.n numbers and store the result in a 64-bit integer.
    // Extract the integer and fractional parts of the multiplication by shifting and masking the result.
    result = (result >> q) + ((result & (qFactor - 1)) >> q);

    // Check if the result overflows the Qm.n format and clip it to the maximum or minimum value if necessary.
    int32_t maxVal = (1 << (m + n - 1)) - 1;
    int32_t minVal = -1 * maxVal - 1;
    if (result > maxVal) {
        result = maxVal;
    }
    else if (result < minVal) {
        result = minVal;
    }

    // Truncate the result to a signed 32-bit integer and return it.
    return (int32_t)result;
}

/**
 * Just your basic 1d matrix multipication.
 * computes C = A*B
 * assumes A,B,C are all 1d arrays of matrices stored in row major order.
 *
 * :param C: output array.
 * :param A: input array 1.
 * :param B: input array 2.
 * :param outrows: number of rows of C and A.
 * :param outcols: number of cols of C and B.
 * :param innderdim: number of cols of A and rows of B
 */
void k2c_matmul(float * C, const float * A, const float * B, const size_t outrows,
                const size_t outcols, const size_t innerdim) {

    // make sure output is empty
    memset(C, 0, outrows*outcols*sizeof(C[0]));

    for (size_t i = 0 ; i < outrows; ++i) {
        const size_t outrowidx = i*outcols;
        const size_t inneridx = i*innerdim;
        for (size_t k = 0; k < innerdim; ++k) {
            for (size_t j = 0;  j < outcols; ++j) {
                C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
            }
        }
    }
}

/**
 * Performs fixed-point matrix multiplication with an affine transformation.
 *
 * @param C           Pointer to the output matrix C (result of multiplication)
 * @param A           Pointer to the input matrix A
 * @param B           Pointer to the input matrix B
 * @param d           Pointer to the affine transformation vector d
 * @param outrows     Number of rows in the output matrix C
 * @param outcols     Number of columns in the output matrix C
 * @param innerdim    Inner dimension (number of columns in matrix A and rows in matrix B)
 * @param shift_factor The number of bits to shift the fixed-point multiplication result
 * @param scale_factor The scaling factor used to convert fixed-point result to floating point
 */
void k2c_affine_matmul_fixed_point(int* C, const int* A, const int* B, const int* d,
    const size_t outrows, const size_t outcols, const size_t innerdim,
    size_t shift_factor, size_t scale_factor) {
    // make sure output is empty

    //long int MAC = 0;
    for (size_t i = 0; i < outrows; ++i) {
        //MAC = 0;
        const size_t outrowidx = i * outcols;
        const size_t inneridx = i * innerdim;
        for (size_t j = 0; j < outcols; ++j) {   
            for (size_t k = 0; k < innerdim; ++k) {
                C[outrowidx + j] += multiplyFixedPoint(A[inneridx + k], B[k * outcols + j], shift_factor, shift_factor);
                // every mixed point multiplication needs to be shifted 16 bits to the right to get the correct scale (MAC << 16)
                // we also need to divide by 2^16 to get the value to floating point (done by shifting 16 to the left)
                //C[outrowidx + j] = (float)(MAC >> 16)/(float)(1<<16);
            }
            C[outrowidx + j] = addFixedPoint(C[outrowidx + j], d[j],shift_factor,shift_factor);
            //C[outrowidx + j] += d[j];
        }
    }
    printf("%d and %d\n", (C[0]) >> shift_factor, (C[1]) >> shift_factor);
}



/**
 * Affine matrix multiplication.
 * computes C = A*B + d, where d is a vector that is added to each
 row of A*B
 * assumes A,B,C are all 1d arrays of matrices stored in row major order
 *
 * :param C: output array.
 * :param A: input array 1.
 * :param B: input array 2.
 * :param d: input array 3.
 * :param outrows: number of rows of C and A.
 * :param outcols: number of cols of C, B and d.
 * :param innderdim: number of cols of A and rows of B
 */
void k2c_affine_matmul(float * C, const float * A, const float * B, const float * d,
                       const size_t outrows,const size_t outcols, const size_t innerdim) {

    // make sure output is empty
    memset(C, 0, outrows*outcols*sizeof(C[0]));

    for (size_t i = 0 ; i < outrows; ++i) {
        const size_t outrowidx = i*outcols;
        const size_t inneridx = i*innerdim;
        for (size_t j = 0;  j < outcols; ++j) {
            for (size_t k = 0; k < innerdim; ++k) {
                C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
            }
            C[outrowidx+j] += d[j];
        }
    }
}


/**
 * Converts subscripts to linear indices in row major order.
 *
 * :param sub: array[ndim] subscript to convert.
 * :param shape: array[ndim] shape of array being indexed.
 * :param ndim: number of dimensions of array being indexed.
 * :return: linear index in row major order.
 */
size_t k2c_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim) {

    size_t idx = 0;
    size_t temp = 0;
    for (size_t i=0; i<ndim; ++i) {
        temp = sub[i];
        for (size_t j=ndim-1; j>i; --j) {
            temp *= shape[j];
        }
        idx += temp;
    }
    return idx;
}


/**
 * Converts linear indices to subscripts in row major order.
 *
 * :param idx: linear index in row major order.
 * :param sub: array[ndim] output subscript.
 * :param shape: array[ndim] shape of array being indexed.
 * :param ndim: number of dimensions of array being indexed.
 */
void k2c_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim) {

    size_t idx2 = idx;
    for (int i=ndim-1; i>=0; --i) {
        sub[i] = idx2%shape[i];
        idx2 /= shape[i];
    }
}


/**
 * Dot product (tensor contraction) between 2 tensors. C=A*B
 *
 * :param C: output tensor.
 * :param A: input tensor 1.
 * :param B: input tensor 2.
 * :param axesA: array[naxes] of axes of A being contracted.
 * :param axesB: array[naxes] of axes of B being contracted.
 * :param naxes: number of axes being contracted from each input.
 * :param normalize: (0,1) whether to L2-normalize samples along the dot product axis before taking the dot product. If set to 1, then the output of the dot product is the cosine proximity between the two samples.
 * :param fwork: array of working space, size(fwork) = size(A) + size(B)
 */
void k2c_dot(k2c_tensor* C, const k2c_tensor* A, const k2c_tensor* B, const size_t * axesA,
             const size_t * axesB, const size_t naxes, const int normalize, float * fwork) {

    size_t permA[K2C_MAX_NDIM];
    size_t permB[K2C_MAX_NDIM];
    size_t prod_axesA = 1;
    size_t prod_axesB = 1;
    size_t free_axesA, free_axesB;
    size_t freeA[K2C_MAX_NDIM];
    size_t freeB[K2C_MAX_NDIM];
    size_t count;
    int isin;
    size_t newshpA[K2C_MAX_NDIM];
    size_t newshpB[K2C_MAX_NDIM];
    const size_t ndimA = A->ndim;
    const size_t ndimB = B->ndim;
    float *reshapeA = &fwork[0];   // temp working storage
    float *reshapeB = &fwork[A->numel];
    size_t Asub[K2C_MAX_NDIM];
    size_t Bsub[K2C_MAX_NDIM];
    // find which axes are free (ie, not being summed over)
    count=0;
    for (size_t i=0; i<ndimA; ++i) {
        isin = 0;
        for (size_t j=0; j<naxes; ++j) {
            if (i==axesA[j]) {
                isin=1;
            }
        }
        if (!isin) {
            freeA[count] = i;
            ++count;
        }
    }
    count=0;
    for (size_t i=0; i<ndimB; ++i) {
        isin = 0;
        for (size_t j=0; j<naxes; ++j) {
            if (i==axesB[j]) {
                isin=1;
            }
        }
        if (!isin) {
            freeB[count] = i;
            ++count;
        }
    }

    // number of elements in inner dimension
    for (size_t i=0; i < naxes; ++i) {
        prod_axesA *= A->shape[axesA[i]];
    }
    for (size_t i=0; i < naxes; ++i) {
        prod_axesB *= B->shape[axesB[i]];
    }
    // number of elements in free dimension
    free_axesA = A->numel/prod_axesA;
    free_axesB = B->numel/prod_axesB;
    // find permutation of axes to get into matmul shape
    for (size_t i=0; i<ndimA-naxes; ++i) {
        permA[i] = freeA[i];
    }
    for (size_t i=ndimA-naxes, j=0; i<ndimA; ++i, ++j) {
        permA[i] = axesA[j];
    }
    for (size_t i=0; i<naxes; ++i) {
        permB[i] = axesB[i];
    }
    for (size_t i=naxes, j=0; i<ndimB; ++i, ++j) {
        permB[i] = freeB[j];
    }



    for (size_t i=0; i<ndimA; ++i) {
        newshpA[i] = A->shape[permA[i]];
    }
    for (size_t i=0; i<ndimB; ++i) {
        newshpB[i] = B->shape[permB[i]];
    }

    // reshape arrays
    for (size_t i=0; i<A->numel; ++i) {
        k2c_idx2sub(i,Asub,A->shape,ndimA);
        for (size_t j=0; j<ndimA; ++j) {
            Bsub[j] = Asub[permA[j]];
        }
        size_t bidx = k2c_sub2idx(Bsub,newshpA,ndimA);
        reshapeA[bidx] = A->array[i];
    }

    for (size_t i=0; i<B->numel; ++i) {
        k2c_idx2sub(i,Bsub,B->shape,ndimB);
        for (size_t j=0; j<ndimB; ++j) {
            Asub[j] = Bsub[permB[j]];
        }
        size_t bidx = k2c_sub2idx(Asub,newshpB,ndimB);
        reshapeB[bidx] = B->array[i];
    }


    if (normalize) {

        float sum;
        float inorm;
        for (size_t i=0; i<free_axesA; ++i) {
            sum = 0;
            for (size_t j=0; j<prod_axesA; ++j) {
                sum += reshapeA[i*prod_axesA + j]*reshapeA[i*prod_axesA + j];
            }
            inorm = 1.0f/sqrtf(sum);
            for (size_t j=0; j<prod_axesA; ++j) {
                reshapeA[i*prod_axesA + j] *= inorm;
            }
        }
        for (size_t i=0; i<free_axesB; ++i) {
            sum = 0;
            for (size_t j=0; j<prod_axesB; ++j) {
                sum += reshapeB[i + free_axesB*j]*reshapeB[i + free_axesB*j];
            }
            inorm = 1.0f/sqrtf(sum);
            for (size_t j=0; j<prod_axesB; ++j) {
                reshapeB[i + free_axesB*j] *= inorm;
            }
        }
    }

    k2c_matmul(C->array, reshapeA, reshapeB, free_axesA,
               free_axesB, prod_axesA);
}

/**
 * Adds bias vector b to tensor A.
 * assumes b is a rank 1 tensor that is added to the last dimension of A.
 *
 * :param A: input tensor. Overwritten with outputs.
 * :param b: bias tensor.
 */
void k2c_bias_add_fixed_point(k2c_tensor_int* A, const k2c_tensor_int* b) {
    int shift_factor = 16;
    int scale_factor = 0;
    for (size_t i = 0; i < A->numel; i += b->numel) {
        for (size_t j = 0; j < b->numel; ++j) {
            //A->array[i+j] += b->array[j];
            A->array[i + j] = addFixedPoint(A->array[i + j], b->array[j], shift_factor, shift_factor);
        }
    }
}
/**
 * Adds bias vector b to tensor A.
 * assumes b is a rank 1 tensor that is added to the last dimension of A.
 *
 * :param A: input tensor. Overwritten with outputs.
 * :param b: bias tensor.
 */
void k2c_bias_add(k2c_tensor* A, const k2c_tensor* b) {

    for (size_t i=0; i<A->numel; i+=b->numel) {
        for (size_t j=0; j<b->numel; ++j) {
            A->array[i+j] += b->array[j];
        }
    }
}


/**
 * Flips a tensor along specified axis.
 * overwrites input with flipped output.
 *
 * :param A: input tensor. Overwritten with outputs.
 * :param axis: axis along which to flip
 */

void k2c_flip(k2c_tensor *A, const size_t axis) {
    const size_t ndim = A->ndim;
    const size_t * shape = A->shape;
    const size_t numel = A->numel;
    size_t sub[K2C_MAX_NDIM] = {0};
    const size_t step = 1;
    size_t k = 0;
    size_t idx = 0;
    float temp;

    size_t reduced_size = 1;
    for (size_t i=axis; i<ndim; ++i) {
        reduced_size *= shape[i];
    }
    const size_t threshold = reduced_size/2;
    const size_t jump = reduced_size;

    while (k<numel) {
        k2c_idx2sub(k, sub, shape, ndim);
        sub[axis] = shape[axis]-sub[axis]-1;
        idx = k2c_sub2idx(sub, shape, ndim);
        temp = A->array[k];
        A->array[k] = A->array[idx];
        A->array[idx] = temp;
        if ((k+step) % jump >= threshold) {
            k = (k + step -threshold + jump);
        }
        else {
            k += step;
        }
    }
}



/**
 * Reads array from csv file.
 *
 * :param filename: file to read from. Assumed comma separated ascii text.
 * :param array_size: how many values to read from the file.
 * :return: pointer to allocated array.
 */
float* k2c_read_array(const char* filename, const size_t array_size) {
    float* ptr = (float*) malloc(array_size * sizeof(float));
    if (!ptr) {
        printf("cannot allocate memory %s \n", filename);
        exit(-1);
    }
    size_t ctr = 0;
    FILE *finp;
    int foo;
    finp = fopen(filename, "r");
    if(NULL == finp) {
        printf("Unable to open file %s \n",filename);
        exit(-1);
    }
    while((!feof(finp)) && (ctr < array_size)) {
        foo = fscanf(finp, "%f,", &ptr[ctr++]);
    }
    fclose(finp);
    return ptr;
}
