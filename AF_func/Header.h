//==========================================
//revisi�n 0.8.2 17-01-2020, 22:20, VS 2017
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
Nombre de la variable de entorno que guardar� una ruta en donde deber�a 
encontrarse archivos de c�digo fuente usados por OpenCL, tales como 
encabezados (.h) o kernels (.cl) la variable de entorno tendr� como valor
algo como:
"C:/Users/user/Documents/Visual Studio 2013/Projects/AF_func"
observar el sentido de las barras oblicuas, tambi�n debe obviarse poner
una al final de la variable, para  que esta pueda ser interpretada y 
manejada correctamente en algunas funciones.*/
#define CL_VAR_ENT "CL_FILES"

class AFire
{
public:
	//-----------------
	//funciones b�sicas
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
		/*multiplicaci�n de matriz sim�trica por vector,
		la matriz debe ser ingresada en formato disperso
		ver funci�n fac_sparse_chol_c */
		void sparse_mat_vec_mul(af_array* dC, af_array elmA,
			af_array colA, af_array rowA, af_array dB);
	static HEADER_API
		/*multiplicaci�n de matriz sim�trica por vector,
		la matriz debe ser ingresada en formato SKS
		(Skyline Storage)
		ver funci�n fac_sparse_chol_sks */
		void sparse_sks_mat_vec_mul(af_array* dC,
			af_array elmA, af_array idxA, af_array dB);
	//-----------------
	//2. Algebra lineal
	//-----------------
	//----------------
	//2.1 Gauss-Jordan
	//----------------
	/**
	soluci�n de un sistema de ecuaciones lineales de la forma:
	Ax=B
	donde x es el vector inc�gnita, los argumentos son
	C: Vector de tama�o n, que contendr� la soluci�n
	A: Matriz cuadrada de orden n
	B: Vector de tama�o n
	*/
	static HEADER_API
		void SELgj_f(af_array*, af_array, af_array);
	static HEADER_API
		/*Soluci�n de un sistema de ecuaciones de la
		forma Ax=b, usando el m�todo de Gauss-Jordan
		x: vector de salida que contendr� la soluci�n
		A: matriz de coeficientes del sistema
		b: Vector de constantes del sistema
		*/
		void SELgj_c(af_array* x, af_array A, af_array b);
	static HEADER_API
		/*Soluci�n de un sistema de ecuaciones de la
		forma Ax=b, usando el m�todo de Gauss-Jordan
		x: vector de salida que contendr� la soluci�n
		Ma: Matriz aumentada del sistema.
		Las operaciones se realizar�n en sitio
		*/
		void SELgj_c(af_array* x, af_array Ma);

	static HEADER_API
		void SELgj_fshr(af_array*, af_array, af_array);
	static HEADER_API
		void SELgj_f2d(af_array*, af_array, af_array);
	static HEADER_API
		void SELgj_c2d(af_array*, af_array, af_array);

	//-------------------------
	//2.2 gradientes conjugados
	//-------------------------

	static HEADER_API
	/*Soluci�n del sistema de ecuaciones lineales de la
	forma Ax=b,	usando el m�todo de gradientes
	conjugados, A es una matriz sim�trica y
	definida positiva.
	sol: contendr� la soluci�n
	Ierr: error m�ximo para la soluci�n
	*/
		void SEL_gc(af_array* sol, af_array A,
			af_array b, double Ierr);
	static HEADER_API
	/*Soluci�n del sistema de ecuaciones lineales de la
	forma Ax=b,	usando el m�todo de gradientes
	conjugados, A es una matriz sim�trica y 
	definida positiva.
	Ierr: error m�ximo para la soluci�n
	Las operaciones se realizar�n en sitio, b
	contendr� al final la soluci�n del sistema.
	*/
		void SEL_gc(af_array A, af_array b,
			double Ierr);
	static HEADER_API
		af::array SEL_gc(af::array, af::array, double);
	static HEADER_API
		/*Soluci�n del sistema de ecuaciones lineales de la
		forma Ax=b para matrices en forma dispersa,
		usando el m�todo de gradientes conjugados

		A debe ser introducido como argumento en su forma
		dispersa (ver funci�n fac_sparse_chol_c),
		siendo A una matriz sim�trica y definida
		positiva*/
		void SELgc_sparse(af_array* C, af_array elmA,
			af_array colA, af_array rowA, af_array b,
			double Ierr);
	static HEADER_API
		/*Soluci�n del sistema de ecuaciones lineales de la
		forma Ax=b para matrices en forma dispersa,
		usando el m�todo de gradientes conjugados

		A debe ser introducido como argumento en su forma
		dispersa (ver funci�n fac_sparse_chol_c),
		siendo A una matriz sim�trica y definida
		positiva
		
		Las operaciones se realizar�n en sitio, 
		se modificar� b, para contener la soluci�n*/
		void SELgc_sparse( af_array elmA,
			af_array colA, af_array rowA, af_array b,
			double Ierr);
	static HEADER_API
		/*Soluci�n del sistema de ecuaciones lineales
		de la forma Ax=b para matrices en forma
		dispersa (SKS), usando el m�todo de gradientes
		conjugados

		A debe ser introducido como argumento en su forma
		dispersa (ver funci�n fac_sparse_chol_sks),
		siendo A una matriz sim�trica y
		definida positiva*/
		void SELgc_sparse_sks(af_array* C,
			af_array elmA, af_array idxA, af_array b,
			double Ierr);
	static HEADER_API
		/*Soluci�n del sistema de ecuaciones lineales
		de la forma Ax=b para matrices en forma
		dispersa (SKS), usando el m�todo de gradientes
		conjugados

		A debe ser introducido como argumento en su forma
		dispersa (ver funci�n fac_sparse_chol_sks),
		siendo A una matriz sim�trica y
		definida positiva.
		
		Las operaciones se realizar�n en sitio, b
		se modificar� para contener la soluci�n*/
		void SELgc_sparse_sks(af_array elmA,
			af_array idxA, af_array b, double Ierr);
	//------------
	//2.3 Cholesky
	//------------
	static HEADER_API
		/*Soluci�n de un sistema de ecuaciones de la
		forma Ax=b, usando la factorizaci�n de cholesky
		x: vector de salida que contendr� la soluci�n
		A: matriz simetrica definida positiva
		b: Vector de constantes del sistema
		*/
		void SELchol_c(af_array* x, af_array A, af_array b);
	static HEADER_API
		/*Soluci�n de un sistema de ecuaciones de la
		forma Ax=b, con A sim�trica y definida positiva.
		L: factor L producto de la factorizaci�n de
		   Cholesky, tal que L*transpose(L)=A
		b: Vector de constantes del sistema
		Se resuelve el sistema en sitio, i.e, b 
		contendr� la soluci�n del sistema al final
		*/
		void SELchol_c(af_array L, af_array b);

	static HEADER_API
		/*Factorizaci�n de cholesky de la matriz A
		sim�trica y definida positiva, la matriz de
		salida tendr� en su parte triangular inferior
		la factorizaci�n tal que:
		L cumple:
		L*tranpose(L)=A*/
		void fac_chol_c(af_array* L, af_array A);
	static HEADER_API
		/*Factorizaci�n de cholesky en sitio 
		de la matriz A sim�trica y definida positiva,
		la matriz tendr� en su parte triangular 
		inferior la factorizaci�n L tal que:
		L*tranpose(L)=A*/
		void fac_chol_c(af_array A);
	static HEADER_API
		/*Copia los elementos de A en el lugar
		correspondiente de L, A y L son matrices
		dipersas con sus tres componentes elm,
		col y row; est� funci�n es especial para 
		copiar los valores del campo de elementos
		de una matriz dispersa A, en el lugar 
		correspondiente del campo de elementos de 
		la matriz dispersa L, est� �ltima producto
		de una factorizaci�n simb�lica.
		El campo de elementos de L deber�a ser mayor
		o igual en longitud al campo de elementos
		de A, tambi�n, inicialmente, deber�a 
		contener solo ceros, ver ayuda de la 
		funci�n fac_sparse_chol_c.
		*/
		void AFire::sparse_copy_A_to_initL(af_array elmA,
			af_array colA, af_array rowA, af_array elmL,
			af_array colL, af_array rowL);
	static HEADER_API
		/*Realiza la factorizaci�n de Cholesky en
		sitio de la matriz A sim�trica y definida
		positiva, el argumento es la misma matriz A
		en forma dispersa (campos elm, col y row). 
		El campo de elementos deber� tener el suficiente 
		tama�o para almacenar la factorizaci�n, i.e.
		puede ser necesario una factorizaci�n 
		simb�lica previa, para determinar el espacio
		adicional respecto al campo de elementos
		inicial de A*/
		void AFire::fac_sparse_chol_c(af_array elmL, af_array colL,
			af_array rowL);
	static HEADER_API
		/*factorizaci�n de Cholesky aplicado a matrices
		dispersas y sim�tricas:
		Sea una matriz representada en su forma densa por A.
		esta matriz en su forma dispersa ser� representada
		por 3 vectores.

		*elmA: Vector de elementos distintos de cero, guardados
		columna a columna.

		*colA: Vector de �ndices, donde cada elemento indica la
		ubicaci�n en elmA del primer elemento distinto de cero en
		cada columna de A

		*rowA: Vector de �ndices en donde se guardan los �ndices
		de fila en A de los elementos en elmA expetuando los elementos
		que coincidan con la diagonal

		ya que la matriz es sim�trica solo se considera la parte
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
		donde se almacenar� los elementos de L de la factorizaci�n
		de Cholesky, elmL debe ser del tama�o adecuado para guardar
		todos los elementos distintos de cero de L, colL y rowL
		deben estar ya con sus valores finales, estos deben
		obtenerse mediante una factorizaci�n simb�lica*/
		void fac_sparse_chol_c(af_array elmA, af_array colIdxA,
			af_array rowIdxA, af_array elmL, af_array colIdxL,
			af_array rowIdxL);

	static HEADER_API
		/*Soluci�n del sistema de ecuaciones lineales de la
		forma Ax=b para matrices en forma dispersa,
		usando la factorizaci�n de cholesky

		A debe ser introducido como argumento en su forma
		dispersa (ver funci�n fac_sparse_chol), tambi�n
		debe proporcionarse la matriz L en forma dispersa
		sin rellenar (es decir con elmL[i]=0 para todo i),
		este obtenido mediante una factorizaci�n simb�lica
		elmA, colA, rowA, elmL, colL y rowL son los mismos
		argumentos que se introducen al usar la funci�n
		fac_sparse_chol_c*/
		void SELchol_sparse_c(af_array* dC, af_array elmA,
			af_array colA, af_array rowA, af_array elmL,
			af_array colL, af_array rowL, af_array b);

	static HEADER_API
		/*Soluci�n del sistema Ax=b, teniendo el factor
		L en forma dispersa obtenido de la factorizaci�n
		de cholesky

		L debe ser introducido como argumento en su forma
		dispersa (ver funci�n fac_sparse_chol)*/
		void SELchol_sparse_c(af_array* dC, af_array elmL,
			af_array colL, af_array rowL, af_array dB);
	static HEADER_API
		/*Soluci�n del sistema Ax=b, teniendo el factor
		L en forma dispersa obtenido de la factorizaci�n
		de cholesky

		L debe ser introducido como argumento en su forma
		dispersa (ver funci�n fac_sparse_chol).
		
		Las operaciones se realizar�n en sitio, b
		contendr� la soluci�n (al final)*/
		void SELchol_sparse_c(af_array elmL,
			af_array colL, af_array rowL, af_array dB);

	static HEADER_API
		/*Implace
		factorizaci�n de cholesky de una matriz
		dispersa y sim�trica, almacenada en formato
		SKS (Skyline Storage)
		Sea una matriz representada en su forma densa por A.
		esta matriz en su forma dispersa en formato SKS ser�
		representada por 2 vectores.

		*elmA: Vector de elementos distintos de cero, guardados
		columna a columna.

		*idxA: Vector de �ndices, donde cada elemento indica la
		ubicaci�n en elmA del primer elemento distinto de cero en
		cada columna de A

		a diferencia del formato SCS (ver m�s arriba) que
		necesita adicionalmente un vector de �ndices
		de fila, en el formato SKS se asume que los elementos
		en elmA correspondientes a una columna est�n ordenados
		uno despues de otro apartir de la diagonal.

		ya que la matriz es sim�trica solo se considera la parte
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
		hasta el �ltimo elemento distinto de cero, incluyendo
		los ceros que haya entre estos.

		al ir avanzando de columna a columna el �ndice de filas
		no puede disminuir, es por eso que en el ejemplo para
		la columna 2 se almacena un cero adicional. Esto para
		que al realizar la factorizaci�n de cholesky o LDLt,
		ya no sea necesario realizar una factorizaci�n
		simb�lica (la estructura se matiene)

		este formato es ideal cuando los elementos distintos
		cero tienden a estar cerca de la diagonal, (e.g una
		matriz banda), en matrices en donde este
		comportamiento no se da, este formato no es el ideal,
		por ejemplo si solo en la primera columna el �ltimo
		elemento es distinto de cero, siguiendo las reglas
		,tendr�an que almacenarse pr�cticamente todos los
		elementos de la matriz.
		*/
		void fac_sparse_chol_sks(af_array elmA,
			af_array idxA);

	static HEADER_API
		/*Soluci�n del sistema Ax=b, para una
		matriz sim�trica y definida positiva,
		usando la factorizaci�n
		de cholesky, teniendo la factorizaci�n de A,
		almacenada en formato SKS*/
		void SELchol_sparse_sks(af_array* dC, af_array elmL,
			af_array idxL, af_array dB);
	static HEADER_API
		/*Soluci�n del sistema Ax=b, para una
		matriz sim�trica y definida positiva,
		usando la factorizaci�n
		de cholesky, teniendo la factorizaci�n de A,
		almacenada en formato SKS.
		Las operaciones se realizar�n en sitio, i.e,
		b contendr� al final la soluci�n*/
		void SELchol_sparse_sks(af_array elmL,
			af_array idxL, af_array dB);
	//--------
	//2.4 LDLT
	//--------
	static HEADER_API
		/*Soluci�n de un sistema de ecuaciones de la
		forma Ax=b, usando la factorizaci�n LDLT
		x: vector de salida que contendr� la soluci�n
		A: matriz simetrica
		b: Vector de constantes del sistema
		*/
		void SELldlt_c(af_array* x, af_array A, af_array b);
	static HEADER_API
		/*Soluci�n de un sistema de ecuaciones de la
		forma Ax=b, con A sim�trica.
		L: factor L producto de la factorizaci�n
		   LDLT, ver fac_ldlt_c
		b: Vector de constantes del sistema
		Se resuelve el sistema en sitio, i.e, b
		contendr� la soluci�n del sistema al final
		*/
		void SELldlt_c(af_array L, af_array b);

	static HEADER_API
		/*Factorizaci�n LDLT de la matriz A
		sim�trica, la matriz de salida contiene
		en la parte triangular inferior los elementos
		de L y D que cumplen:
		L*D*tranpose(L)=A*/
		void fac_ldlt_c(af_array* L, af_array A);
	static HEADER_API
		/*Factorizaci�n LDLT en sitio
		de la matriz A sim�trica,
		la matriz tendr� en su parte triangular
		inferior los elementos L y D de 
		la factorizaci�n tal que:
		L*D*tranpose(L)=A*/
		void fac_ldlt_c(af_array A);
	static HEADER_API
		/*Realiza la factorizaci�n LDLT en
		sitio de la matriz A sim�trica,
		el argumento es la misma matriz A
		en forma dispersa (campos elm, col y row).
		El campo de elementos deber� tener el suficiente
		tama�o para almacenar la factorizaci�n, i.e.
		puede ser necesario una factorizaci�n
		simb�lica previa, para determinar el espacio
		adicional respecto al campo de elementos
		inicial de A*/
		void AFire::fac_sparse_ldlt_c(af_array elmL, af_array colL,
			af_array rowL);
	static HEADER_API
		/*factorizaci�n LDLt aplicado a matrices
		dispersas y sim�tricas:

		para ver el formato de una matriz almacenada en
		forma dispersa y sim�trica ver fac_sparse_chol_c

		En las funciones L es otra matriz con el mismo formato
		donde se almacenar� los elementos de L de la factorizaci�n
		LDLt, elmL debe ser del tama�o adecuado para guardar
		todos los elementos distintos de cero de L, colL y rowL
		deben estar ya con sus valores finales, estos deben
		obtenerse mediante una factorizaci�n simb�lica,
		tomar en cuenta que la factoriaci�n simb�lica LDLt
		es la misma que la factorizaci�n de cholesky
		simb�lica*/
		void fac_sparse_ldlt_c(af_array elmA, af_array colA,
			af_array rowA, af_array elmL, af_array colL,
			af_array rowL);

	static HEADER_API
		/*Soluci�n del sistema de ecuaciones lineales de la
		forma Ax=b para matrices en forma dispersa,
		usando la factorizaci�n LDLt

		A debe ser introducido como argumento en su forma
		dispersa (ver funci�n fac_sparse_chol), tambi�n
		debe proporcionarse la matriz L en forma dispersa
		sin rellenar (es decir con elmL[i]=0 para todo i),
		este obtenido mediante una factorizaci�n simb�lica
		elmA, colA, rowA, elmL, colL y rowL son los mismos
		argumentos que se introducen al usar la funci�n
		fac_sparse_ldlt_c*/
		void SELldlt_sparse_c(af_array* dC, af_array elmA,
			af_array colA, af_array rowA, af_array elmL,
			af_array colL, af_array rowL, af_array b);

	static HEADER_API
		/*Soluci�n del sistema Ax=b, teniendo el factor
		L en forma dispersa obtenido de la factorizaci�n
		LDLt

		L debe ser introducido como argumento en su forma
		dispersa (ver funci�n fac_sparse_chol)*/
		void SELldlt_sparse_c(af_array* dC, af_array elmL,
			af_array colL, af_array rowL, af_array dB);
	static HEADER_API
		/*Soluci�n del sistema Ax=b, teniendo el factor
		L en forma dispersa obtenido de la factorizaci�n
		LDLT

		L debe ser introducido como argumento en su forma
		dispersa (ver funci�n fac_sparse_ldlt).

		Las operaciones se realizar�n en sitio, b
		contendr� la soluci�n (al final)*/
		void SELldlt_sparse_c(af_array elmL,
			af_array colL, af_array rowL, af_array dB);

	static HEADER_API
		/*Implace
		factorizaci�n LDLt de una matriz
		dispersa y sim�trica, almacenada en formato
		SKS (Skyline Storage)
		ver ayuda de la funci�n fac_sparse_chol_sks
		*/
		void fac_sparse_ldlt_sks(af_array elmA,
			af_array idxA);

	static HEADER_API
		/*Soluci�n del sistema Ax=b, para una
		matriz sim�trica usando la factorizaci�n
		LDLt, teniendo la factorizaci�n de A,
		almacenada en formato SKS*/
		void SELldlt_sparse_sks(af_array* dC, af_array elmL,
			af_array idxL, af_array dB);
	static HEADER_API
		/*Soluci�n del sistema Ax=b, para una
		matriz sim�trica usando la factorizaci�n
		LDLT, teniendo la factorizaci�n de A,
		almacenada en formato SKS.
		Las operaciones se realizar�n en sitio, i.e,
		b contendr� al final la soluci�n*/
		void SELldlt_sparse_sks(af_array elmL,
			af_array idxL, af_array dB);
	//---------------------------
	//3 funciones de prueba y otros
	//---------------------------
	static HEADER_API void test_1(af::array &, af::array &);
	static HEADER_API void test_2(af_array*, af_array, af_array);
	static HEADER_API
		void prueba_shr(af_array*, af_array, af_array);
	static HEADER_API
		void fenceTest_sp(af::array &, af::array &);
	static HEADER_API double SELgj_c_test(af_array,
		af_array);
	static HEADER_API
		double AFire::SELchol_sparse_sks_time(af_array elm,
		af_array Idx, af_array F);
	static HEADER_API
		double AFire::sumaaf_time(af_array A, af_array B);
	static HEADER_API
		double AFire::af_add_time(af_array A, af_array B);

	static HEADER_API
		void global_sync_test(af_array*, af_array,
			af_array);
	static HEADER_API
		void AFire::sks_util_1(af_array* index, af_array row,
			af_array col);
	static HEADER_API
		void AFire::sks_util_2(af_array sks, af_array ptr,
			af_array nz,af_array row, af_array col);
	static HEADER_API
		void AFire::csc_util_1(af_array* index, af_array row,
			af_array col);
	static HEADER_API
		void AFire::csc_util_2(af_array* del, af_array index,
			af_array row);

	static HEADER_API
		void AFire::csc_util_3(af_array del, 
			af_array ptr);

	static HEADER_API
		void AFire::csc_util_4(af_array csc, af_array rowc,
			af_array ptr, af_array nz, af_array row,
			af_array col);
};

