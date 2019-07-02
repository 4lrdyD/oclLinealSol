//==========================================
//revisión 0.7.2 02-07-2019, 00:00, VS 2017
//==========================================
#pragma once
#ifdef HEADER_EXPORTS
#define HEADER_API __declspec(dllexport)
#else
#define HEADER_API __declspec(dllimport)
#endif

#include <arrayfire.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <af/opencl.h>
#include <oclUtils.h>
#include <shrQATest.h>
#include <iostream>

/*
Nombre de la variable de entorno que guardará una ruta en donde debería 
encontrarse archivos de código fuente usados por OpenCL, tales como 
encabezados (.h) o kernels (.cl) la variable de entorno tendrá como valor
algo como:
"C:/Users/user/Documents/Visual Studio 2013/Projects/AF_func"
observar el sentido de las barras oblicuas, también debe obviarse poner
una al final de la variable, para  que esta pueda ser interpretada y 
manejada correctamente en algunas funciones.*/
#define CL_VAR_ENT "CL_FILES"

class AFire
{
public:
	//-----------------
	//funciones básicas
	//-----------------
	static HEADER_API
		void copy(const af::array &, af::array &, size_t);
	static HEADER_API
		void printh(float*, size_t);
	static HEADER_API
		void sumaaf(af_array*, af_array, af_array);
	static HEADER_API
		void mulaf(af_array*, const af_array, const af_array);
	static HEADER_API
		char * find_kernel_test(const char *);
	static HEADER_API
		char * helpk(const char *);
	static HEADER_API
		char * get_env_path(const char *);
	static HEADER_API
		char * kernel_src(const char*, const char*);

	static HEADER_API
		/*multiplicación de matriz simétrica por vector,
		la matriz debe ser ingresada en formato disperso
		ver función fac_sparse_chol_c */
		void sparse_mat_vec_mul(af_array* dC, af_array elmA,
			af_array colA, af_array rowA, af_array dB);
	//--------------
	//Algebra lineal
	//--------------
	/**
	solución de un sistema de ecuaciones lineales de la forma:
	Ax=B
	donde x es el vector incógnita, los argumentos son
	C: Vector de tamaño n, que contendrá la solución
	A: Matriz cuadrada de orden n
	B: Vector de tamaño n
	*/
	static HEADER_API 
		void SELgj_f(af_array*, af_array, af_array);
	static HEADER_API 
		/*Solución de un sistema de ecuaciones de la
		forma Ax=b, usando el método de Gauss-Jordan
		x: vector de salida que contendrá la solución
		A: matriz de coeficientes del sistema
		b: Vector de constantes del sistema
		*/
		void SELgj_c(af_array* x, af_array A, af_array b);
	static HEADER_API 
		void SELgj_fshr(af_array*, af_array, af_array);
	static HEADER_API 
		void SELgj_f2d(af_array*, af_array, af_array);
	static HEADER_API 
		void SELgj_c2d(af_array*, af_array, af_array);
	static HEADER_API
		void prueba_shr(af_array*, af_array, af_array);
	static HEADER_API 
		void fenceTest_sp(af::array &, af::array &);
	//---------------------
	//gradientes conjugados
	//---------------------
	static HEADER_API
		void SEL_gc(af_array*, af_array, af_array, double);
	static HEADER_API
		af::array SEL_gc(af::array, af::array, double);
   	static HEADER_API
		/*Solución del sistema de ecuaciones lineales de la
		forma Ax=b para matrices en forma dispersa,
		usando el método de gradientes conjugados

		A debe ser introducido como argumento en su forma
		dispersa (ver función fac_sparse_chol), siendo la
		forma densa una matriz simétrica y definida 
		positiva*/
	void SELgc_sparse(af_array* C, af_array elmA,
		af_array colA, af_array rowA, af_array b,
		double Ierr);
	//--------
	//Cholesky
	//--------
	static HEADER_API
		/*Solución de un sistema de ecuaciones de la 
		forma Ax=b, usando la factorización de cholesky
		x: vector de salida que contendrá la solución
		A: matriz simetrica definida positiva 
		b: Vector de constantes del sistema
		*/
		void SELchol_c(af_array* x, af_array A, af_array b);

	static HEADER_API
		/*Factorización de cholesky de la matriz A
		simétrica y definida positiva, la matriz de salida
		L cumple:
		L*tranpose(L)=A*/
		void fac_chol_c(af_array* L, af_array A);

	static HEADER_API
		/*factorización de Cholesky aplicado a matrices
		dispersas y simétricas:
		Sea una matriz representada en su forma densa por A.
		esta matriz en su forma dispersa será representada
		por 3 vectores.

		*elmA: Vector de elementos distintos de cero, guardados
		columna a columna.

		*colA: Vector de índices, donde cada elemento indica la 
		ubicación en elmA del primer elemento distinto de cero en 
		cada columna de A

		*rowA: Vector de índices en donde se guardan los índices 
		de fila en A de los elementos en elmA expetuando los elementos
		que coincidan con la diagonal

		ya que la matriz es simétrica solo se considera la parte
		triangular inferior.

		ejemplo para:
		A=[	5	1  -2   0
		1	2	0	0
		-2	0	4	1
		0	0	1	3]

		elmA=[	5
				1
				-2
				2
				4
				1
				3]

		colA=[	0
				3
				4
				6]

		rowA=[	1
				2
				3]
	

		En las funciones L es otra matriz con el mismo formato
		donde se almacenará los elementos de L de la factorización
		de Cholesky, elmL debe ser del tamaño adecuado para guardar
		todos los elementos distintos de cero de L, colL y rowL
		deben estar ya con sus valores finales, estos deben
		obtenerse mediante una factorización simbólica*/
	void fac_sparse_chol_c(af_array elmA, af_array colIdxA,
		af_array rowIdxA, af_array elmL, af_array colIdxL,
		af_array rowIdxL);
	
	static HEADER_API
		/*Solución del sistema de ecuaciones lineales de la
		forma Ax=b para matrices en forma dispersa,
		usando la factorización de cholesky

		A debe ser introducido como argumento en su forma
		dispersa (ver función fac_sparse_chol), también
		debe proporcionarse la matriz L en forma dispersa
		sin rellenar (es decir con elmL[i]=0 para todo i),
		este obtenido mediante una factorización simbólica
		elmA, colA, rowA, elmL, colL y rowL son los mismos
		argumentos que se introducen al usar la función
		fac_sparse_chol_c*/
		void SELchol_sparse_c(af_array* dC, af_array elmA,
			af_array colA, af_array rowA, af_array elmL,
			af_array colL, af_array rowL, af_array b);

	static HEADER_API
		/*Solución del sistema Ax=b, teniendo el factor
		L en forma dispersa obtenido de la factorización 
		de cholesky

		L debe ser introducido como argumento en su forma
		dispersa (ver función fac_sparse_chol)*/
	void SELchol_sparse_c(af_array* dC, af_array elmL,
		af_array colL, af_array rowL, af_array dB);
	
	static HEADER_API
		/*Implace
		factorización de cholesky de una matriz
		dispersa y simétrica, almacenada en formato
		SKS (Skyline Storage)
		Sea una matriz representada en su forma densa por A.
		esta matriz en su forma dispersa en formato SKS será
		representada por 2 vectores.

		*elmA: Vector de elementos distintos de cero, guardados
		columna a columna.

		*idxA: Vector de índices, donde cada elemento indica la
		ubicación en elmA del primer elemento distinto de cero en
		cada columna de A

		a diferencia del formato SCS (ver más arriba) que
		necesita adicionalmente un vector de índices
		de fila, en el formato SKS se asume que los elementos
		en elmA correspondientes a una columna están ordenados
		uno despues de otro apartir de la diagonal.

		ya que la matriz es simétrica solo se considera la parte
		triangular inferior.

		ejemplo para:
		A=[	3  4  0  1  0  0  0
			4  1  2  0  0  0  0
			0  2  2  4  4  0  0
			1  0  4  9  0  5  0
			0  0  4  0  2  0  1
			0  0  0  5  0  5  0
			0  0  0  0  1  0  8]

		elmA=[	3
				4
				0
				1
				1
				2
				0
				2
				4
				4
				9
				0	
				5
				2
				0
				1
				5
				0
				8]

		idxA=[	0
				4
				7
				10
				13
				16
				18]

		para la columna i el rango de sus elementos en elmA
		van de index[i] a index[i+1]-1

		se almacenan todos los elementos desde la diagonal
		hasta el último elemento distinto de cero, incluyendo
		los ceros que haya entre estos.

		al ir avanzando de columna a columna el índice de filas
		no puede disminuir, es por eso que en el ejemplo para
		la columna 2 se almacena un cero adicional. Esto para
		que al realizar la factorización de cholesky o LDLt,
		ya no sea necesario realizar una factorización
		simbólica (la estructura se matiene)

		este formato es ideal cuando los elementos distintos
		cero tienden a estar cerca de la diagonal, (e.g una
		matriz banda), en matrices en donde este
		comportamiento no se da, este formato no es el ideal,
		por ejemplo si solo en la primera columna el último
		elemento es distinto de cero, siguiendo las reglas
		,tendrían que almacenarse prácticamente todos los
		elementos de la matriz.
		*/
		void fac_sparse_chol_sks(af_array elmA,
			af_array idxA);

	static HEADER_API
		/*Solución del sistema Ax=b, para una
		matriz simétrica usando la factorización
		de cholesky, teniendo la factorización de A,
		almacenada en formato SKS*/
		void SELchol_sparse_sks(af_array* dC, af_array elmL,
			af_array idxL, af_array dB);
	//----
	//LDLT
	//----
	static HEADER_API
		/*Solución de un sistema de ecuaciones de la
		forma Ax=b, usando la factorización LDLT
		x: vector de salida que contendrá la solución
		A: matriz simetrica
		b: Vector de constantes del sistema
		*/
		void SELldlt_c(af_array* x, af_array A, af_array b);

	static HEADER_API
		/*Factorización LDLT de la matriz A
		simétrica, la matriz de salida contiene
		en la parte triangular inferior los elementos
		de L y D que cumplen:
		L*D*tranpose(L)=A*/
		void fac_ldlt_c(af_array* L, af_array A);

	static HEADER_API
		/*factorización LDLt aplicado a matrices
		dispersas y simétricas:
		
		para ver el formato de una matriz almacenada en 
		forma dispersa y simétrica ver fac_sparse_chol_c 
		
		En las funciones L es otra matriz con el mismo formato
		donde se almacenará los elementos de L de la factorización
		LDLt, elmL debe ser del tamaño adecuado para guardar
		todos los elementos distintos de cero de L, colL y rowL
		deben estar ya con sus valores finales, estos deben
		obtenerse mediante una factorización simbólica, 
		tomar en cuenta que la factoriación simbólica LDLt
		es la misma que la factorización de cholesky 
		simbólica*/
	void fac_sparse_ldlt_c(af_array elmA, af_array colA,
		af_array rowA, af_array elmL, af_array colL,
		af_array rowL);

	static HEADER_API
		/*Solución del sistema de ecuaciones lineales de la
		forma Ax=b para matrices en forma dispersa,
		usando la factorización LDLt

		A debe ser introducido como argumento en su forma
		dispersa (ver función fac_sparse_chol), también
		debe proporcionarse la matriz L en forma dispersa
		sin rellenar (es decir con elmL[i]=0 para todo i),
		este obtenido mediante una factorización simbólica
		elmA, colA, rowA, elmL, colL y rowL son los mismos
		argumentos que se introducen al usar la función
		fac_sparse_ldlt_c*/
		void SELldlt_sparse_c(af_array* dC, af_array elmA,
			af_array colA, af_array rowA, af_array elmL,
			af_array colL, af_array rowL, af_array b);

	static HEADER_API
		/*Solución del sistema Ax=b, teniendo el factor
		L en forma dispersa obtenido de la factorización
		LDLt

		L debe ser introducido como argumento en su forma
		dispersa (ver función fac_sparse_chol)*/
		void SELldlt_sparse_c(af_array* dC, af_array elmL,
			af_array colL, af_array rowL, af_array dB);
	
	static HEADER_API
		/*Implace
		factorización LDLt de una matriz 
		dispersa y simétrica, almacenada en formato
		SKS (Skyline Storage)
		ver ayuda de la función fac_sparse_chol_sks
        */
	void fac_sparse_ldlt_sks(af_array elmA,
		af_array idxA);

	static HEADER_API
		/*Solución del sistema Ax=b, para una
		matriz simétrica usando la factorización
		LDLt, teniendo la factorización de A, 
		almacenada en formato SKS*/
	void SELldlt_sparse_sks(af_array* dC, af_array elmL,
		af_array idxL, af_array dB);
	//-------------------
	//funciones de prueba
	//-------------------
	static HEADER_API void test_1(af::array &, af::array &);
	static HEADER_API void test_2(af_array);
};

