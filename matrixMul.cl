/*
=====================================
revisi�n 0.3 01-02-2019, 20:40 VS 2017
======================================
con algunos cambios en base al c�digo fuente de ejemplo
incorporado en OpenCL, compatible con cualquier orden de
matriz, no necesariamente multiplos de BLOCK_SIZE
*/
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */
#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]

///////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! uiWA is A's width and uiWB is B's width
////////////////////////////////////////////////////////////////////////////////
//----------------
//double presicion
//----------------
__kernel void
matrixMul(__global double* C, __global double* A, __global double* B,
	__local double* As, __local double* Bs, int uiWA, int uiWB, int trueLocalSize1)
{
	// Block index
	int bx = get_group_id(0);
	int by = get_group_id(1);

	// Thread index
	int tx = get_local_id(0);
	int ty = get_local_id(1);

	// Index of the first sub-matrix of A processed by the block
	int aBegin = uiWA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + uiWA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * uiWB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	double Csub = 0.0;

	//indice general de un elemento en C
	int IDX = BLOCK_SIZE * bx + tx;
	int IDY = BLOCK_SIZE * by + ty;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix

		AS(ty, tx) =
			(aEnd >= a + tx && IDY < trueLocalSize1 ?
				A[a + uiWA * ty + tx] : 0);

		BS(ty, tx) =
			(aEnd >= a + ty && IDX < uiWB ?
				B[b + uiWB * ty + tx] : 0);

		// Synchronize to make sure the matrices are loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix        
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += AS(ty, k) * BS(k, tx);

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	int c = uiWB * BLOCK_SIZE * by +
		BLOCK_SIZE * bx;
	// Write the block sub-matrix to device memory;
	// each thread writes one element
	if (IDX < uiWB && IDY < trueLocalSize1)
		C[c + uiWB * ty + tx] = Csub;
}

//----------------
//single presicion
//----------------
__kernel void
matrixMul_sp(__global float* C, __global float* A, __global float* B,
	__local float* As, __local float* Bs, int uiWA, int uiWB, int trueLocalSize1)
{
	// Block index
	int bx = get_group_id(0);
	int by = get_group_id(1);

	// Thread index
	int tx = get_local_id(0);
	int ty = get_local_id(1);

	// Index of the first sub-matrix of A processed by the block
	int aBegin = uiWA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + uiWA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * uiWB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0.0;

	//indice general de un elemento en C
	int IDX = BLOCK_SIZE * bx + tx;
	int IDY = BLOCK_SIZE * by + ty;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix

		AS(ty, tx) =
			(aEnd >= a + tx && IDY < trueLocalSize1 ?
				A[a + uiWA * ty + tx] : 0);

		BS(ty, tx) =
			(aEnd >= a + ty && IDX < uiWB ?
				B[b + uiWB * ty + tx] : 0);

		// Synchronize to make sure the matrices are loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix        
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += AS(ty, k) * BS(k, tx);

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	int c = uiWB * BLOCK_SIZE * by +
		BLOCK_SIZE * bx;
	// Write the block sub-matrix to device memory;
	// each thread writes one element
	if (IDX < uiWB && IDY < trueLocalSize1)
		C[c + uiWB * ty + tx] = Csub;
}