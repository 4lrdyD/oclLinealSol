/*
=========================================
revisi�n 0.9.04 09-04-2019, 17:15 VS 2017
=========================================
*/
/* Soluci�n de un sistema de ecuaciones, usando 
el m�todo de Gauss_Jordan: para un sistema de la forma:
A*x=B donde la inc�gnita es x; A es una matriz cuadrada
de orden n y B es un vector de longitud n
los argumentos ser�n modificados, al final Ac contendr�
en su �ltima columna y en su diagonal principal, los 
valores finales que deben dividirse para obtener la 
soluci�n final 
 */
#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]

void __gpu_sync(int goalVal,
	__global volatile int * Arrayin, __global volatile int * Arrayout);

void __gpu_sync1(int goalVal,
	__global volatile int * g_mutex);

void Gss_Jrd_f_sp(__global float* A, int wA, int hA,
	int row, int col, int frow, int fcol);

void Gss_Jrd_f(__global double* A, int wA, int hA,
	int row, int col, int frow, int fcol);

void Gss_Jrd_c_sp(__global float* A, int wA, int hA,
	int row, int col, int frow, int fcol);

void Gss_Jrd_c(__global double* A, int wA, int hA,
	int row, int col, int frow, int fcol);

void Gss_Jrd_cshr_sp(__global float* A, __local float* colBs,
	__local float* colEls, int wA, int hA, int row, int col,
	int frow, int fcol);

//------------------------
//simple precisi�n (float)
//------------------------

/*Kernel que concibe los elementos de A, como ordenados por fila
simple precisi�n*/
__kernel void
Gauss_Jordan_f_sp(__global float* A, int hA, int row)
{
	//la iteraci�n se realizar� en el host, en aqu�
	//solo se relizara una operaci�n sobre cada fila
	//encargada a cada hilo
	Gss_Jrd_f_sp(A, hA + 1, hA, row, row, 0, row + 1);
}

/*
funci�n llamada por el kernel Gauss_Jordan_f_sp
Resta a cada fila de A la fila "row" multiplicada por un
factor, tal que el elemento ubicado en la posici�n "col" 
de cada fila sea cero luego de realizada la operaci�n,
"fcol" indica la posici�n en cada fila desde donde se 
realizar�n las operaciones, si es para la fila completa 
el valor correspondiente de "fcol" es cero, del mismo modo
debe especificarse una fila (frow) para obviar las filas
anteriores en las operaciones, para todas las filas el 
valor es cero.

Se asume A como una matriz de orden wAxhA
tambi�n se asume que la cantidad de bloques en el kernel
que llamar� esta funci�n es suficiente

Se asume la matriz como ordenada por filas
*/
void Gss_Jrd_f_sp(__global float* A, int wA, int hA,
	int row, int col, int frow, int fcol)
{
	// frow + �ndice local de grupo
	int bx = frow + get_group_id(0);

	//�ndice del primer elemento en una fila
	int tx = fcol + get_local_id(0);
	
	//factor de multiplicaci�n
	float fm;

	//fila base, la que se restar� a todas las filas
	const __global float* rowB = A + row * wA;

	//en este caso, cada grupo se encargar� de las operaciones sobre todos los
	//elementos de una fila

	float keyEl = rowB[col];

	for (int y = bx; y < hA; y += get_num_groups(0)) {

		if (y != row && keyEl != 0) {

			__global float* rowEl = A + y * wA;
			fm = native_divide(rowEl[col], keyEl);

			for (int x = tx; x < wA; x += get_local_size(0)) {
				rowEl[x] -= fm* rowB[x];
			}
		}
	}
}

/*Kernel que concibe los elementos de A, como ordenados por columna
simple precisi�n
*/
__kernel void
Gauss_Jordan_c_sp(__global float* A, int hA, int row)
{
	//la iteraci�n se realizar� en el host, en aqu�
	//solo se relizara una operaci�n sobre cada columna
	//encargada a cada hilo
	Gss_Jrd_c_sp(A, hA + 1, hA, row, row, 0, row + 1);
}

/*
funci�n llamada por el kernel Gauss_Jordan_c_sp
Resta a cada fila de A la fila "row" multiplicada por un
factor, tal que el elemento ubicado en la posici�n "col"
de cada fila sea cero luego de realizada la operaci�n,
"fcol" indica la posici�n en cada fila desde donde se
realizar�n las operaciones, si es para la fila completa
el valor correspondiente de "fcol" es cero, del mismo modo
debe especificarse una fila (frow) para obviar las filas
anteriores en las operaciones, para todas las filas el
valor es cero.

Se asume A como una matriz de orden wAxhA
tambi�n se asume que la cantidad de bloques en el kernel
que llamar� esta funci�n es suficiente

se asume la matriz como ordenada por columnas
*/
void Gss_Jrd_c_sp(__global float* A, int wA, int hA,
	int row, int col, int frow, int fcol)
{
	// fcol + �ndice local de grupo
	int bx = fcol + get_group_id(0);

	//�ndice del primer elemento en una columna
	int tx = frow + get_local_id(0);

	//factor de multiplicaci�n
	float fm;

	//columna base, la que se vuelve cero, excepto el
	//valor en la posici�n frow
	__global float* Colb = A + hA * col;
    

	//en este caso, cada grupo se encargar� de las operaciones sobre todos los
	//elementos de una columna
	float keyEl = Colb[row];

	for (int x = tx; x < hA; x += get_local_size(0)) {

		if (x != row && keyEl != 0) {

			fm = native_divide(Colb[x], keyEl);
			for (int y = bx; y < wA; y += get_num_groups(0)) {

				__global float* ColEl = A + y * hA;
				ColEl[x] -= fm * ColEl[row];
			}
		}
	}
}

/*Kernel que concibe los elementos de A, como ordenados por fila
usando memoria compartida (precisi�n simple)*/
__kernel void
Gauss_Jordan_fshr_sp(__global float* C, __global float* A,
	__global float* B,__local float * As, int gsize,
	__global volatile int * syncIn, __global volatile int * syncOut)
{
	// �ndice local de hilo
	int tlx = get_local_id(0);
	int tly = get_local_id(1);

	// �ndice global de hilo
	int tgx = get_global_id(0);
	int tgy = get_global_id(1);

	//�ndice del primer elemento en una fila
	int Rfirst = tgy * gsize;

	//elemento diagonal
	float diag;

	//factor de multiplicaci�n
	float fm;

	//valor necesario para la sincronizaci�n
	int syncStep= get_num_groups(0)*get_num_groups(1);
	int syncGoal = syncStep;

	//cargando una submatriz de A
	As[tlx + tly * get_local_size(0)] = (tgx < gsize && tgy < gsize ?
		A[Rfirst + tgx] : 0);

	//sincronizando para asegurar que todas los elementos de As
	//fueron llenados correctamente
	//barrier(CLK_LOCAL_MEM_FENCE);

	//en este caso, cada hilo se encargar� de las operaciones sobre su elemento
	//correspondiente
	
	for (int i = 0; i < gsize; i++)
	{
		diag = A[i*gsize + i];

		if (tgx < gsize && tgy < gsize &&
			tgy != i && diag != 0)
		{
			fm = native_divide(A[Rfirst + i], diag);
			if (tgx > i)
				As[tlx + tly * get_local_size(0)] -= fm * A[i*gsize + tgx];
			else if (tgx == i)
				B[tgy] -= fm * B[i];
			else;

		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (tgx < gsize && tgy < gsize)
			A[Rfirst + tgx] = As[tlx + tly * get_local_size(0)];

		//sincronizando para asegurar que todas las filas fueron cambiadas,
		//antes de continuar la iteraci�n
		__gpu_sync1(syncGoal, syncIn);
		//__gpu_sync(syncGoal, syncIn, syncOut);
		syncGoal += syncStep;
	}

	//hasta aqu� solo nos interesa la diagonal de A
	if (tgx == 0 && tgy < gsize)
	{
		C[tgy] = native_divide(B[tgy], A[tgy*gsize + tgy]);
	}
}

/*
funci�n llamada por el kernel Gauss_Jordan_c_sp
Resta a cada fila de A la fila "row" multiplicada por un
factor, tal que el elemento ubicado en la posici�n "col"
de cada fila sea cero luego de realizada la operaci�n,
"fcol" indica la posici�n en cada fila desde donde se
realizar�n las operaciones, si es para la fila completa
el valor correspondiente de "fcol" es cero, del mismo modo
debe especificarse una fila (frow) para obviar las filas
anteriores en las operaciones, para todas las filas el
valor es cero.

Se asume A como una matriz de orden wAxhA
tambi�n se asume que la cantidad de bloques en el kernel
que llamar� esta funci�n es suficiente

se asume la matriz como ordenada por columnas
*/
void Gss_Jrd_cshr_sp(__global float* A, __local float* colBs,
	__local float* colEls, int wA, int hA, int row, int col,
	int frow, int fcol)
{
	// fcol + �ndice local de grupo
	int bx = fcol + get_group_id(0);

	//�ndice local de hilo
	int tlx = get_local_id(0);

	//�ndice del primer elemento en una columna
	int tx = frow + tlx;

	//factor de multiplicaci�n
	float fm;

	//columna base, la que se vuelve cero, excepto el
	//valor en la posici�n frow
	__global float* Colb = A + hA * col;

	//en este caso, cada grupo se encargar� de las operaciones sobre todos los
	//elementos de una columna
	float keyEl = Colb[row];

	for (int x = tx; x < hA; x += get_local_size(0)) {
    //cargando ColBs
		colBs[tlx] = Colb[x];

		if (x != row && keyEl != 0) {

			fm = native_divide(colBs[tlx], keyEl);
			for (int y = bx; y < wA; y += get_num_groups(0)) {

				__global float* ColEl = A + y * hA;

				//cargando colEls
				colEls[tlx] = ColEl[x];

				colEls[tlx] -= fm * ColEl[row];

				//devolviendo los valores
				ColEl[x] = colEls[tlx];
			}
		}
	}
}

/*Kernel que concibe los elementos de A como ordenados por fila,
usando grupos de hilo bidimensional, precisi�n simple*/
__kernel void
Gauss_Jordan_f2d_sp(__global float* C, __global float* A,
	__global float* B, int gsize)
{
	// �ndice local de hilo y
	int tlx = get_local_id(0);
	int tly = get_local_id(1);

	// �ndice de grupo
	int bx = get_group_id(0);
	int by = get_group_id(1);

	//�ndice del primer elemento en una fila
	int Rfirst;

	//elemento diagonal
	float diag;

	//factor de multiplicaci�n
	float fm;

	//�ndice referencial
	int tgx;

	/*en este caso, cada hilo se encargar� de las operaciones sobre su elemento
	correspondiente y el elemento situado a BLOCK_SIZE filas abajo*/

	int step = BLOCK_SIZE;
	for (int i = 0; i < gsize; i++)
	{
		diag = A[i*gsize + i];

		for (int k = 0; k < gsize; k += step)
		{
			Rfirst = (tly + k)*gsize;
			fm = native_divide(A[Rfirst + i], diag);

			for (int j = 0; j < gsize; j += step)
			{
				if (tlx + j < gsize && tly + k < gsize &&
					tly + k != i && diag != 0)
				{
					tgx = tlx + j;
					if (i < tgx)
						A[Rfirst + tgx] -= fm * A[i*gsize + tgx];
					else if (tgx == i)
						B[tly + k] -= fm * B[i];
					else;
				}
			}
		}
		/*sincronizando para asegurar que todas las filas fueron cambiadas,
		antes de continuar la iteraci�n*/
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/*hasta aqu� solo nos interesa la diagonal de A*/
	for (int k = 0; k < gsize; k += step)
	{
		if (tlx == 0 && tly + k < gsize)
		{
			C[tly + k] = native_divide(B[tly + k],
				A[(tly + k)*gsize + tly + k]);
		}
	}
}

/*Kernel que concibe los elementos de A como ordenados por columna,
usando grupos de hilo bidimensional, precisi�n simple*/
__kernel void
Gauss_Jordan_c2d_sp(__global float* C, __global float* A,
	__global float* B, int gsize)
{
	// �ndice local de hilo y
	int tlx = get_local_id(0);
	int tly = get_local_id(1);

	// �ndice de grupo
	int bx = get_group_id(0);
	int by = get_group_id(1);

	//�ndice del primer elemento en una fila
	int Rfirst;

	//elemento diagonal
	float diag;

	//factor de multiplicaci�n
	float fm;

	//�ndice referencial
	int tgx;

	/*en este caso, cada hilo se encargar� de las operaciones sobre su elemento
	correspondiente y el elemento situado a BLOCK_SIZE filas abajo*/

	int step = BLOCK_SIZE;
	for (int i = 0; i < gsize; i++)
	{
		diag = A[i*gsize + i];

		for (int k = 0; k < gsize; k += step)
		{
			Rfirst = tly + k;
			fm = native_divide(A[Rfirst + i * gsize], diag);

			for (int j = 0; j < gsize; j += step)
			{
				if (tlx + j < gsize && tly + k < gsize &&
					tly + k != i && diag != 0)
				{
					tgx = tlx + j;
					if (i < tgx)
						A[Rfirst + tgx * gsize] -= fm * A[i + tgx * gsize];
					else if (tgx == i)
						B[tly + k] -= fm * B[i];
					else;
				}
			}
		}
		/*sincronizando para asegurar que todas las filas fueron cambiadas,
		antes de continuar la iteraci�n*/
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/*hasta aqu� solo nos interesa la diagonal de A*/
	for (int k = 0; k < gsize; k += step)
	{
		if (tlx == 0 && tly + k < gsize)
		{
			C[tly + k] = native_divide(B[tly + k],
				A[(tly + k)*gsize + tly + k]);
		}
	}
}

//------------------------
//doble precisi�n (double)
//------------------------
//Kernel que concibe los elementos de A, como ordenados por fila
__kernel void
Gauss_Jordan_f(__global double* A,
	int hA, int row) 
{
	//la iteraci�n se realizar� en el host, en aqu�
	//solo se relizara una operaci�n sobre cada fila
	//encargada a cada hilo
	Gss_Jrd_f(A, hA + 1, hA, row, row, 0, row + 1);
}

/*
funci�n llamada por el kernel Gauss_Jordan_f_sp
Resta a cada fila de A la fila "row" multiplicada por un
factor, tal que el elemento ubicado en la posici�n "col"
de cada fila sea cero luego de realizada la operaci�n,
"fcol" indica la posici�n en cada fila desde donde se
realizar�n las operaciones, si es para la fila completa
el valor correspondiente de "fcol" es cero, del mismo modo
debe especificarse una fila (frow) para obviar las filas
anteriores en las operaciones, para todas las filas el
valor es cero.

Se asume A como una matriz de orden wAxhA
tambi�n se asume que la cantidad de bloques en el kernel
que llamar� esta funci�n es suficiente
*/
void Gss_Jrd_f(__global double* A, int wA, int hA,
	int row, int col, int frow, int fcol)
{
	// frow + �ndice local de grupo
	int bx = frow + get_group_id(0);

	//�ndice del primer elemento en una fila
	int tx = fcol + get_local_id(0);

	//factor de multiplicaci�n
	double fm;

	//fila base, la que se restar� a todas las filas
	const __global double* rowB = A + row * wA;

	//en este caso, cada grupo se encargar� de las operaciones sobre todos los
	//elementos de una fila

	double keyEl = rowB[col];

	for (int y = bx; y < hA; y += get_num_groups(0)) {

		if (y != row && keyEl != 0) {

			__global double* rowEl = A + y * wA;
			fm = rowEl[col] / keyEl;

			for (int x = tx; x < wA; x += get_local_size(0)) {
				rowEl[x] -= fm * rowB[x];
			}
		}
	}
}

//Kernel que concibe los elementos de A, como ordenados por columna
__kernel void
Gauss_Jordan_c(__global double* A,
	int hA, int row)
{
	//la iteraci�n se realizar� en el host, en aqu�
	//solo se relizara una operaci�n sobre cada columna
	//encargada a cada hilo
	Gss_Jrd_c(A, hA + 1, hA, row, row, 0, row + 1);
}

/*
funci�n llamada por el kernel Gauss_Jordan_c_sp
Resta a cada fila de A la fila "row" multiplicada por un
factor, tal que el elemento ubicado en la posici�n "col"
de cada fila sea cero luego de realizada la operaci�n,
"fcol" indica la posici�n en cada fila desde donde se
realizar�n las operaciones, si es para la fila completa
el valor correspondiente de "fcol" es cero, del mismo modo
debe especificarse una fila (frow) para obviar las filas
anteriores en las operaciones, para todas las filas el
valor es cero.

Se asume A como una matriz de orden wAxhA
tambi�n se asume que la cantidad de bloques en el kernel
que llamar� esta funci�n es suficiente

se asume la matriz como ordenada por columnas
*/
void Gss_Jrd_c(__global double* A, int wA, int hA,
	int row, int col, int frow, int fcol)
{
	// fcol + �ndice local de grupo
	int bx = fcol + get_group_id(0);

	//�ndice del primer elemento en una columna
	int tx = frow + get_local_id(0);

	//factor de multiplicaci�n
	double fm;

	//columna base, la que se vuelve cero, excepto el
	//valor en la posici�n frow
	__global double* Colb = A + hA * col;


	//en este caso, cada grupo se encargar� de las operaciones sobre todos los
	//elementos de una columna
	double keyEl = Colb[row];

	for (int x = tx; x < hA; x += get_local_size(0)) {

		if (x != row && keyEl != 0) {

			fm = Colb[x] / keyEl;
			for (int y = bx; y < wA; y += get_num_groups(0)) {

				__global double* ColEl = A + y * hA;
				ColEl[x] -= fm * ColEl[row];
			}
		}
	}
}

/*Kernel que concibe los elementos de A, como ordenados por fila
usando memoria compartida*/
__kernel void
Gauss_Jordan_fshr(__global double* C, __global double* A, __global double* B,
	__local double As[BLOCK_SIZE][BLOCK_SIZE], int gsize)
{
	// �ndice local de hilo
	int tlx = get_local_id(0);
	int tly = get_local_id(1);

	// �ndice global de hilo
	int tgx = get_global_id(0);
	int tgy = get_global_id(1);

	//�ndice del primer elemento en una fila
	int Rfirst = tgy * gsize;

	//elemento diagonal
	double diag;

	//factor de multiplicaci�n
	double fm;

	//cargando una submatriz de A
	As[tly][tlx] = (tgx < gsize && tgy < gsize ?
		A[Rfirst + tgx] : 0);

	/*sincronizando para asegurar que todas los elementos de As
	fueron llenados correctamente*/
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	/*en este caso, cada hilo se encargar� de las operaciones sobre su elemento
	correspondiente*/

	for (int i = 0; i < gsize; i++)
	{
		diag = A[i*gsize + i];

		if (tgx < gsize && tgy < gsize &&
			tgy != i && diag != 0)
		{
			fm = A[Rfirst + i] / diag;
			if (tgx > i)
				As[tly][tlx] -= fm * A[i*gsize + tgx];
			else if (tgx == i)
				B[tgy] -= fm * B[i];
			else;

		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (tgx < gsize && tgy < gsize)
			A[Rfirst + tgx] = As[tly][tlx];

		/*sincronizando para asegurar que todas las filas fueron cambiadas,
		antes de continuar la iteraci�n*/
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}

	/*hasta aqu� solo nos interesa la diagonal de A*/
	if (tgx == 0 && tgy < gsize)
	{
		C[tgy] = B[tgy] / A[tgy*gsize + tgy];
	}
}

/*Kernel que concibe los elementos de A como ordenados por fila,
usando grupos de hilo bidimensional*/
__kernel void
Gauss_Jordan_f2d(__global double* C, __global double* A,
	__global double* B, int gsize)
{
	// �ndice local de hilo y
	int tlx = get_local_id(0);
	int tly = get_local_id(1);

	// �ndice de grupo
	int bx = get_group_id(0);
	int by = get_group_id(1);

	//�ndice del primer elemento en una fila
	int Rfirst;

	//elemento diagonal
	double diag;

	//factor de multiplicaci�n
	double fm;

	//�ndice referencial
	int tgx;

	/*en este caso, cada hilo se encargar� de las operaciones sobre su elemento
	correspondiente y el elemento situado a BLOCK_SIZE filas abajo*/

	int step = BLOCK_SIZE;
	for (int i = 0; i < gsize; i++)
	{
		diag = A[i*gsize + i];

		for (int k = 0; k < gsize; k += step)
		{
			Rfirst = (tly + k)*gsize;
			fm = A[Rfirst + i] / diag;

			for (int j = 0; j < gsize; j += step)
			{
				if (tlx + j < gsize && tly + k < gsize &&
					tly + k != i && diag != 0)
				{
					tgx = tlx + j;
					if (i < tgx)
						A[Rfirst + tgx] -= fm * A[i*gsize + tgx];
					else if (tgx == i)
						B[tly + k] -= fm * B[i];
					else;
				}
			}
		}
		/*sincronizando para asegurar que todas las filas fueron cambiadas,
		antes de continuar la iteraci�n*/
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/*hasta aqu� solo nos interesa la diagonal de A*/
	for (int k = 0; k < gsize; k += step)
	{
		if (tlx == 0 && tly + k < gsize)
		{
			C[tly + k] = B[tly + k] /
				A[(tly + k)*gsize + tly + k];
		}
	}
}

/*Kernel que concibe los elementos de A como ordenados por columna,
usando grupos de hilo bidimensional, precisi�n doble*/
__kernel void
Gauss_Jordan_c2d(__global double* C, __global double* A,
	__global double* B, int gsize)
{
	// �ndice local de hilo y
	int tlx = get_local_id(0);
	int tly = get_local_id(1);

	// �ndice de grupo
	int bx = get_group_id(0);
	int by = get_group_id(1);

	//�ndice del primer elemento en una fila
	int Rfirst;

	//elemento diagonal
	double diag;

	//factor de multiplicaci�n
	double fm;

	//�ndice referencial
	int tgx;

	/*en este caso, cada hilo se encargar� de las operaciones sobre su elemento
	correspondiente y el elemento situado a BLOCK_SIZE filas abajo*/

	int step = BLOCK_SIZE;
	for (int i = 0; i < gsize; i++)
	{
		diag = A[i*gsize + i];

		for (int k = 0; k < gsize; k += step)
		{
			Rfirst = tly + k;
			fm = A[Rfirst + i * gsize] / diag;

			for (int j = 0; j < gsize; j += step)
			{
				if (tlx + j < gsize && tly + k < gsize &&
					tly + k != i && diag != 0)
				{
					tgx = tlx + j;
					if (i < tgx)
						A[Rfirst + tgx * gsize] -= fm * A[i + tgx * gsize];
					else if (tgx == i)
						B[tly + k] -= fm * B[i];
					else;
				}
			}
		}
		/*sincronizando para asegurar que todas las filas fueron cambiadas,
		antes de continuar la iteraci�n*/
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/*hasta aqu� solo nos interesa la diagonal de A*/
	for (int k = 0; k < gsize; k += step)
	{
		if (tlx == 0 && tly + k < gsize)
		{
			C[tly + k] = B[tly + k] /
				A[(tly + k)*gsize + tly + k];
		}
	}
}

//------------------------
//c�digo de sincronizaci�n
//------------------------
/*C�digo realizado con referencia al art�culo:
Inter-Block GPU Communication via Fast Barrier Synchronization
Shucai Xiao and Wu-chun Feng
Department of Computer Science Virginia Tech
2009/9/19
Fig.9. P�g.6
*/

//GPU lock - free synchronization function
void __gpu_sync(int goalVal,
	__global volatile int * Arrayin, __global volatile int * Arrayout)
{
	//thread ID in a block
	int tid_in_block = get_local_id(0)*get_local_size(1)
		+ get_local_id(1);

	int nBlockNum = get_num_groups(0)*get_num_groups(1);

	int bid = get_group_id(0)*get_num_groups(1)
		+ get_group_id(1);
	
	//only thread 0 is used for synchronization
	if (tid_in_block == 0){
		Arrayin[bid] = goalVal; 
	}

	if (bid == 0){
		if (tid_in_block < nBlockNum){
			while (Arrayin[tid_in_block] != goalVal) {
				//atomicCAS();
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (tid_in_block < nBlockNum) {
			Arrayout[tid_in_block] = goalVal;
		}
	}
	if (tid_in_block == 0) {
		while (Arrayout[bid] != goalVal) {
			//atomicCAS();
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

/*C�digo realizado con referencia al art�culo:
Inter-Block GPU Communication via Fast Barrier Synchronization
Shucai Xiao and Wu-chun Feng
Department of Computer Science Virginia Tech
2009/9/19
Fig.6. P�g.5
*/

//GPU simple synchronization function
void __gpu_sync1(int goalVal,
	__global volatile int * g_mutex)
{
	//thread ID in a block
	int tid_in_block = get_local_id(0)*get_local_size(1)
		+ get_local_id(1);

	//only thread 0 is used for synchronization
	if (tid_in_block == 0) {
		atom_inc(&(g_mutex[0]));

		//only when all blocks add 1 to g_mutex will
		//g_mutex equal to goalVal
		while (g_mutex[0] != goalVal) {

		}
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

/*Kernel que concibe los elementos de A, como ordenados por fila
usando memoria compartida*/
__kernel void
prueba_shr(__global double* C, __global double* A, __global double* B,
	__local double As[BLOCK_SIZE][BLOCK_SIZE], int gsize)
{
	// �ndice local de hilo
	int tlx = get_local_id(0);
	int tly = get_local_id(1);

	// �ndice global de hilo
	int tgx = get_global_id(0);
	int tgy = get_global_id(1);

	if (tgx >= gsize || tgy >= gsize)
	{
		return;
	}
	//�ndice del primer elemento en una fila
	int Rfirst = tgy * gsize;

	//elemento diagonal
	double diag;

	//factor de multiplicaci�n
	double fm;


	A[Rfirst + tgx] = tlx*10+tly;

}

/*Kernel que concibe los elementos de A, como ordenados por fila
usando memoria compartida*/
__kernel void
prueba_shr_sp(__global float* C, __global float* A,
	__global volatile float* B, __local float * As,
	__local float * Bs, int gsize)
{
	// �ndice local de hilo
	int tlx = get_local_id(0);
	int tly = get_local_id(1);

	//�ndice de grupo
	int bx = get_group_id(0);
	int by = get_group_id(1);

	// �ndice global de hilo
	int tgx = get_global_id(0);
	int tgy = get_global_id(1);
	//transpuesta
	/*
	if (tgx < gsize && tgy < gsize)
	{
		if (bx > by)
		{
			AS(tly, tlx) = A[tgx*gsize + tgy];
			BS(tlx, tly) = A[tgy*gsize + tgx];
			barrier(CLK_LOCAL_MEM_FENCE);
			A[tgy*gsize + tgx] = AS(tly, tlx);
			A[tgx*gsize + tgy] = BS(tlx, tly);

		}
		else if (bx == by)
		{
			AS(tly, tlx) = A[tgx*gsize + tgy];
			barrier(CLK_LOCAL_MEM_FENCE);
			A[tgy*gsize + tgx] = AS(tly, tlx);
		}
		else;
	}*/
	if (tlx == 0 && tly==0) {

		//atom_add(&(C[0]), 1);
	}
}

/*Prueba2, mem_fence*/
__kernel void fenceTest_sp(__global float *c,
	__global int *ctr, int gsize)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);

	if (tx == 0 && ty == 0) {
		bool isSet = false;
		do
		{
			if (isSet = atom_cmpxchg(&(ctr[0]), 0, 1) == 0)
			{
				ctr[1] += 1;
			}
			if (isSet)
			{
				atom_xchg(&(ctr[0]), 0);
			}
		} while (!isSet);
		/*bool leaveLoop = false;
		while (!leaveLoop) {
			if (atom_xchg(&(ctr[0]), 1) == 0) {
				ctr[1] += 1;
				leaveLoop = true;
				atom_xchg(&(ctr[0]), 0);
			}
		}
		/*while (atom_cmpxchg(&(ctr[0]), 0, 1) != 0) {
		}
		ctr[1] += 1;
		atom_xchg(&(ctr[0]), 0);*/
	}
	if (tx == 0)
		c[ty] = ctr[1];
}