//=======================================
//revisión 0.3.1 20-06-2019, 21:35 VS 2017
//=======================================

//tamaño de bloque para ser usado en funciones normales y Kernels
#define BLOCK_SIZE 32

//activando doble precisión (double) para kernels
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable