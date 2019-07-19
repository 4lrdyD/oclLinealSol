//=======================================
//revisi�n 0.3.2 18-07-2019, 22:40 VS 2017
//=======================================

//tama�o de bloque para ser usado en funciones normales y Kernels
#define BLOCK_SIZE 32

//activando doble precisi�n (double) para kernels
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

//activando funciones at�micas para 32 y 64 bits
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable