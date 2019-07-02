/*
=========================================
revisi�n 0.0.8 02-07-2019, 00:00 VS 2017
=========================================
*/
/* Soluci�n de un sistema de ecuaciones, usando 
la factorizaci�n de Cholesky: para un sistema de la forma:
A*x=B donde la inc�gnita es x; A es una matriz cuadrada
sim�trica y definida positiva de orden n y B es un vector
de longitud n, una vez hecha la factorizaci�n ser�
necesario resolver un sistema equivalente teniendo 
la matriz triangular generada por la factorizaci�n

Ver:
*T�cnicas de C�lculo para Sistemas de Ecuaciones,
Programaci�n Lineal y Programaci�n Entera
Jos� Luis de la Fuente O�Connor
Cap.1, 1.5. Factorizaci�n de matrices sim�tricas
1.5.2 Factorizaci�n de Cholesky
P�g. 40

*FACTORIZACI�N DE CHOLESKY MODIFICADA DE MATRICES DISPERSAS
SOBRE MULTIPROCESADORES
Mar�a J. Mart�n Santamar�a
Cap.2, 2.4. Factorizaci�n num�rica
2.4.1. Factorizaci�n de Cholesky est�ndar
P�g. 14
 */
void __gpu_sync(int goalVal,
	__global volatile int * Arrayin, __global volatile int * Arrayout);

void __gpu_sync1(int goalVal,
	__global volatile int * g_mutex);

//float
void Chol_1_csp(__global float* A, int order, int step);
void Chol_2_csp(__global float* A, int order, int step);
void Chol_3_csp(__global float* A, int order);
void Chol_4_csp(__global float* L, __global float* b, int order,
	int step);
void Chol_5_csp(__global float* L, __global float* b, int order);
void Chol_6_csp(__global float* L, __global float* y, int order,
	int step, __local float* partialSum);

void sparse_fill_csp(__global float* elmA, __global int* colA,
	__global int* rowA, __global float* elmL, __global int* colL,
	__global int* rowL, int size_elmA);

void Chol_sparse_1_csp(__global float* elmL, __global int* colL,
	int order, int step);

void Chol_sparse_2_csp(__global float* elmL, __global int* colL,
	__global int* rowL, int order, int step);

void Chol_sparse_3_csp(__global float* elmL,
	__global int* colL, int order);

void Chol_sparse_4_csp(__global float* elmL,
	__global int* colL, __global int* rowL, __global float* b,
	int order, int step);

void Chol_sparse_5_csp(__global float* elmL,
	__global int* colL, __global float* b, int order);

void Chol_sparse_6_csp(__global float* elmL,
	__global int* colL, __global int* rowL, __global float* y,
	int order, int step, __local float* partialSum);

void chol_sparse_1_sks_sp(__global float* elmL,
	__global int* idxL, int order, int step);

void chol_sparse_2_sks_sp(__global float* elmL,
	__global int* idxL, int order, int step);

void chol_sparse_3_sks_sp(__global float* elmL,
	__global int* idxL, int order);

void chol_sparse_4_sks_sp(__global float* elmL,
	__global int* idxL, __global float* b, int order,
	int step);

void chol_sparse_5_sks_sp(__global float* elmL,
	__global int* idxL, __global float* b, int order);

void chol_sparse_6_sks_sp(__global float* elmL,
	__global int* idxL, __global float* y,
	int order, int step, __local float* partialSum);

//double
void Chol_1_c(__global double* A, int order, int step);
void Chol_2_c(__global double* A, int order, int step);
void Chol_3_c(__global double* A, int order);
void Chol_4_c(__global double* L, __global double* b, int order,
	int step);
void Chol_5_c(__global double* L, __global double* b, int order);
void Chol_6_c(__global double* L, __global double* y, int order,
	int step, __local double* partialSum);
void sparse_fill(__global double* elmA, __global int* colA,
	__global int* rowA, __global double* elmL, __global int* colL,
	__global int* rowL, int size_elmA);

void Chol_sparse_1(__global double* elmL, __global int* colL,
	int order, int step);

void Chol_sparse_2(__global double* elmL, __global int* colL,
	__global int* rowL, int order, int step);

void Chol_sparse_3(__global double* elmL,
	__global int* colL, int order);

void Chol_sparse_4(__global double* elmL,
	__global int* colL, __global int* rowL, __global double* b,
	int order, int step);

void Chol_sparse_5(__global double* elmL,
	__global int* colL, __global double* b, int order);

void Chol_sparse_6(__global double* elmL,
	__global int* colL, __global int* rowL, __global double* y,
	int order, int step, __local double* partialSum);

void chol_sparse_1_sks(__global double* elmL,
	__global int* idxL, int order, int step);

void chol_sparse_2_sks(__global double* elmL,
	__global int* idxL, int order, int step);

void chol_sparse_3_sks(__global double* elmL,
	__global int* idxL, int order);

void chol_sparse_4_sks(__global double* elmL,
	__global int* idxL, __global double* b, int order,
	int step);

void chol_sparse_5_sks(__global double* elmL,
	__global int* idxL, __global double* b, int order);

void chol_sparse_6_sks(__global double* elmL,
	__global int* idxL, __global double* y,
	int order, int step, __local double* partialSum);

//------------------------
//simple precisi�n (float)
//------------------------

/*Kernel que concibe los elementos de A, como ordenados por 
columna simple precisi�n*/
__kernel void
Cholesky_c_sp(__global float* A, __global float* b,
	__local float* partialSum, int order, int step, int sstep)
{
	//la iteraci�n principal se realiza en el host

	//se modificar� una sola columna
	if (sstep == 0) {
		Chol_1_csp(A, order, step);
	}
	//se modificar� las columnas siguientes a la columna
	//implicada
	else if (sstep==1) {
		Chol_2_csp(A, order, step);
	}

	//una vez usada las funciones anteriores todav�a es 
	//necesario modificar la diagonal D[i]=sqrt(D[i]) para 
	//concluir la factorizaci�n
	else if (sstep==2){
		Chol_3_csp(A, order);
	}

	///hasta aqui la factorizaci�n esta concluida, la parte
	//triangular inferior de A contendr� la factorizaci�n de
	//cholesky es decir L
	//desde aqui se proceder� a solucionar el sistema 
	//L*transpose(L)x=b

	//se resolver� el sistema Ly=b, y=transpose(L)*x
	else if (sstep==3){
		Chol_4_csp(A, b, order, step);
	}
	else if (sstep == 4) {
		Chol_5_csp(A, b, order);
	}

	//se resolver� el sistema transpose(L)*x=y
	//"y" ya debe haberse obtenido usando la funci�n anterior
	else{
		Chol_6_csp(A, b, order, step, partialSum);
	}
}

/*
paso j(1) de la factorizaci�n de Cholesky in situ (es decir
la matriz A ser� modificada) en este paso, se modifica
la columna j.

Se modifican los elementos de la columna j a partir de la fila
j + 1.

A(n,j)=A(n,j)/sqrt(A(j,j)) n>j

se asume un almacenamiento por columna

Ver:
FACTORIZACI�N DE CHOLESKY MODIFICADA DE MATRICES DISPERSAS 
SOBRE MULTIPROCESADORES
Mar�a J. Mart�n Santamar�a
Cap.2, 2.4. Factorizaci�n num�rica 
2.4.1. Factorizaci�n de Cholesky est�ndar
P�g. 14
*/
void Chol_1_csp(__global float* A, int order, int step)
{

	//step + 1 + �ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//columna base
	__global float* colB = A + step * order;

	//raiz cuadrada del elemento A(step,step)
	float _r = native_sqrt(colB[step]);

	//cada hilo se encargar� de modificar un elemento 
	//de la columna
	for (int x = tx + step + 1; x < order; x += ntx) {
		colB[x] = native_divide(colB[x], _r);
	}
}

/*
paso j(2) de la factorizaci�n de Cholesky in situ (es decir
la matriz A ser� modificada) para este paso, se modifica
las columnas siguientes a la columna j.

para una columna s, s>j, se modifican las filas n>=s

A(n,s)=A(n,s)-A(s,j)*A(n,j) s>j, n>=s

se asume un almacenamiento por columna

Ver:
FACTORIZACI�N DE CHOLESKY MODIFICADA DE MATRICES DISPERSAS
SOBRE MULTIPROCESADORES
Mar�a J. Mart�n Santamar�a
Cap.2, 2.4. Factorizaci�n num�rica
2.4.1. Factorizaci�n de Cholesky est�ndar
P�g. 14
*/
void Chol_2_csp(__global float* A, int order, int step)
{
	//�ndice local de grupo
	int bx = get_group_id(0);

	//step + �ndice local de hilo
	int tx = get_local_id(0);

	//columna base
	__global float* colB = A + step * order;

	//par�metros de apoyo
	float _k;

	//en este caso, cada grupo se encargar� de las 
	//operaciones sobre todos los elementos de una 
	//columna
	for (int y = bx + step + 1; y < order; y += get_num_groups(0)) {
		_k = colB[y];
		__global float* colI = A + y * order;

		for (int x = tx + y; x < order;
			x += get_local_size(0)) {

			colI[x] -= _k * colB[x];
		}
	}
}

/*
Modifica la diagonal de A 
A(j,j)=sqrt(A(j,j))

se asume un almacenamiento por columna

Ver:
FACTORIZACI�N DE CHOLESKY MODIFICADA DE MATRICES DISPERSAS
SOBRE MULTIPROCESADORES
Mar�a J. Mart�n Santamar�a
Cap.2, 2.4. Factorizaci�n num�rica
2.4.1. Factorizaci�n de Cholesky est�ndar
P�g. 14
*/
void Chol_3_csp(__global float* A, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//parametros de apoyo
	int diag;

	for (int x = tx; x < order; x += ntx) {
		//ubicaci�n del elemento diagonal en memoria
		diag = x * order + x;
		A[diag] = native_sqrt(A[diag]);
	}
}

/*
Inicia la soluci�n del sistema Ax=b teniendo la matriz trangular
inferior L (obtenido tras la factorizaci�n de cholesky), 
tal que A= L*transpose(L)

Con esta funci�n se iniciar� la soluci�n del sistema Ly=b con
y=transpose(L)*x que ser� resuelto posteriormente con la funci�n
Chol_5_csp

para un paso j, modificar� el vector b tal que:

b[n]=b[n]-b[j]*L[n,j]/L[j,j] n>j

este proceso es equivalente a usar el m�todo de Gauss_Jordan
en matrices triangulares inferiores, en este caso es posible 
dejar intacta la matriz L modificando solo el vector b, a pesar
de que para un paso j la idea central es convertir en cero toda
esa columna, estas operaciones pueden obviarse ya que no se 
necesita el valor final, por lo tanto lo que importa solo es el
trabajo que se realiza sobre el vector b

se asume un almacenamiento por columna
*/
void Chol_4_csp(__global float* L, __global float* b, int order,
	int step)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//columna de L implicada
	__global float* colB = L + order * step;

	//elemeto diagonal
	float diag = colB[step];

	float k = native_divide(b[step], diag);

	for (int x = tx + step + 1; x < order; x += ntx) {
		b[x] -= k * colB[x];
	}
}

/*
divide el vector b, entre la diagonal de L
b[j]=b[j]/L[j,j]
*/
void Chol_5_csp(__global float* A, __global float* b, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//parametros de apoyo
	int diag;

	for (int x = tx; x < order; x += ntx) {
		//ubicaci�n del elemento diagonal en memoria
		diag = x * order + x;
		b[x] = native_divide(b[x], A[diag]);
	}
}

/*
Inicia la soluci�n del sistema Ax=b teniendo la matriz trangular
inferior L (obtenido tras la factorizaci�n de cholesky),
tal que A= L*transpose(L)

Con esta funci�n se iniciar� la soluci�n del sistema 
transpose(L)*x=y, "y" debe haberse obtenido anteriormente usando
la funci�n Chol_4_csp (y= b/diag(L), al final de usar Chol_4_csp).
para un paso j modificar� el elemento b[i] tal que i=n-j:

b[i]=(b[i]-sumatoria{j=i+1,n}(y[j]*L[j,i]))/L[i,i]

este valor ser� la soluci�n final x[i] del sistema Ax=b

se asume un almacenamiento por fila, esto para poder usar la
misma matriz L sin necesidad de transponerla, adem�s se usar�
solo un bloque.
*/
void Chol_6_csp(__global float* L, __global float* y, int order,
	int step, __local float* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//�ndice global de hilo
	int tx = get_local_id(0);

	//n�mero total de hilos
	int ntx = get_local_size(0);

	//indice de la fila de L implicada
	int fL = order - step - 1;

	//fila de L implicada, data
	__global float* filB = L + fL * order;

	//elemento diagonal
	float diag = filB[fL];

	//valor donde se guardar� una suma parcial
	float sum = 0.0f;

	//solo se usar� un bloque
	if (bx == 0) {
		for (int x = tx + fL + 1; x < order; x += ntx) {
			sum += filB[x] * y[x];
		}
		partialSum[tx] = sum;

		//reducci�n en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = ntx / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			y[fL] = native_divide(y[fL] - partialSum[0], diag);
	}
}


/*Aplicado a matrices dispersas y sim�tricas:

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

//-------------------------------
__kernel void
Cholesky_sparse_c_sp(__global float* elmA,__global int* colA,
	__global int* rowA, __global float* elmL, __global int* colL,
	__global int* rowL, int size_elmA, int size_colA,
	__global float* b, __local float* partialSum, int step,
	int sstep)
{

	if (sstep == -1)
		sparse_fill_csp(elmA, colA, rowA, elmL, colL,
			rowL, size_elmA);
	//la iteraci�n principal se realiza en el host

		//se modificar� una sola columna
	else if (sstep == 0)
		Chol_sparse_1_csp(elmL, colL, size_colA, step);

	//se modificar� las columnas siguientes a la columna
	//implicada
	else if (sstep == 1)
		Chol_sparse_2_csp(elmL, colL, rowL, size_colA, step);

	//una vez usada las funciones anteriores todav�a es 
	//necesario modificar la diagonal D[i]=sqrt(D[i]) para 
	//concluir la factorizaci�n
	else if (sstep == 2)
		Chol_sparse_3_csp(elmL, colL, size_colA);
	//hasta aqui la factorizaci�n esta concluida, la parte
	//triangular inferior de A contendr� la factorizaci�n de
	//cholesky es decir L
	//desde aqui se proceder� a solucionar el sistema 
	//L*transpose(L)x=b

	//se resolver� el sistema Ly=b, y=transpose(L)*x
	else if (sstep == 3)
		Chol_sparse_4_csp(elmL, colL, rowL, b, size_colA, step);
	else if (sstep == 4)
		Chol_sparse_5_csp(elmL, colL, b, size_colA);

	//se resolver� el sistema transpose(L)*x=y
	//"y" ya debe haberse obtenido usando la funci�n anterior
	else
		Chol_sparse_6_csp(elmL, colL, rowL, b, size_colA, step,
			partialSum);
}

/*dada dos matrices dispersas A y L, copia los elementos de
elmA en el lugar correspondiente de elmL.
Se asume que los �ndices de rowA est�n incluidos en 
rowL. Es decir, en su forma densa, todos
los lugares diferentes de cero en A, tambi�n son diferentes de
cero en L, pero no viceversa*/

void sparse_fill_csp(__global float* elmA, __global int* colA,
	__global int* rowA, __global float* elmL, __global int* colL,
	__global int* rowL, int size_elmA)
{
	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < size_elmA; x += ntx) {
		//variable donde se guardar� el �ndice de columna y de fila
		//respectivamente del elemento elmA[tx]
		int col = 0;
		int row = 0;
		//variable que controlar� la salida del bucle.
		int key = 0;

		while (key == 0) {
			if (colA[col] > x) {
				col -= 1;
				row = rowA[x - col - 1];
				key = 1;
			}
			else if (colA[col] == x) {
				row = col;
				key = 1;
			}
			else
				col += 1;
		}
		//hasta aqui para un elemento de ElmA, ya se
		//se cuenta con el �ndice de fila y de columna (row y col)
		//ahora necesitamos ubicar este elemento en elmL
		
		int posf;//posici�n final en elmL

		if (col != row) {
			int pos = colL[col] - col;
			int cont = 0;
			key = 0;
			while (key == 0)
				rowL[pos + cont] == row ? key = 1 : cont++;
			posf = colL[col] + cont + 1;
		}
		else
			posf = colL[col];
		//escribiendo elmentos de elmA en el lugar 
		//correspondiente de elmL
		elmL[posf] = elmA[x];
	}
}

void Chol_sparse_1_csp(__global float* elmL, __global int* colL,
	int order, int step)
{
	//es procedimiento solo deber� realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la �ltima 
	//columna tendr� �ndice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1){
		//obteniendo la ubicaci�n del primer elemento 
		//de la columna determinada por step
		int pos = colL[step];

		//posici�n siguiente
		int pos_nxt = colL[step + 1];

		//�ndice global de hilo
		int tx = get_global_id(0);

		//n�mero total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//raiz cuadrada del elemento diagonal. El primer
		//elemento guardado en elmA para cada columna es el
		//elemento diagonal
		float _r = native_sqrt(elmL[pos]);

		//cada hilo se encargar� de modificar un elemento 
		//de la columna
		for (int x = tx + pos + 1; x < pos_nxt; x += ntx) {
			elmL[x] = native_divide(elmL[x], _r);
		}
	}
}

void Chol_sparse_2_csp(__global float* elmL, __global int* colL,
	__global int* rowL, int order, int step)
{
	//es procedimiento solo deber� realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la �ltima 
	//columna tendr� �ndice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//�ndice local de grupo
		int bx = get_group_id(0);

		//�ndice local de hilo
		int tx = get_local_id(0);

		//posici�n edl primer elemento de la columna de elmL
		int pos_elmL = colL[step];

		//posici�n del primer elemento de la columna en rowL
		int pos_rowL = pos_elmL - step;

		//n�mero de elementos en rowL para la columna
		int nEl = colL[step + 1] - pos_elmL - 1;

		if (nEl > 0) {

			for (int b = bx; b < nEl; b += get_num_groups(0)) {
				//indice de la columna a modificar
				int colId = rowL[pos_rowL + b];

				//primer factor
				float k1 = elmL[pos_elmL + b + 1];

				for (int x = b + tx; x < nEl; x += get_local_size(0)) {

					//�ndice de la fila en la columna a modificar
					int rowId = rowL[pos_rowL + x];

					//segundo factor
					float k2 = elmL[pos_elmL + x + 1];

					int posf;//posici�n final en elmL

					if (colId != rowId) {
						int pos = colL[colId] - colId;
						int cont = 0;
						int key = 0;
						while (key == 0)
							rowL[pos + cont] == rowId ? key = 1 : cont++;
						posf = colL[colId] + cont + 1;
					}
					else
						posf = colL[colId];
					
					elmL[posf] -= k1 * k2;
				}
			}
		}
	}
}

void Chol_sparse_3_csp(__global float* elmL,
	__global int* colL, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		int diag = colL[x];
		elmL[diag] = native_sqrt(elmL[diag]);
	}
}

void Chol_sparse_4_csp(__global float* elmL,
	__global int* colL, __global int* rowL, __global float* b,
	int order, int step)
{
	if (step < order - 1) {
		//�ndice global de hilo
		int tx = get_global_id(0);

		//n�mero total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//posici�n del elemento diagonal
		int diag = colL[step];

		//n�mero de elementos para una columna
		int nEl = colL[step + 1] - diag;

		//deben haber elementos aparte del elemento diagonal
		if (nEl > 1) {

			//elemento diagonal y elemento correspondiente en b
			float diag_elm = elmL[diag];
			float elmb = b[step];

			//posici�n en rowL del primer elemento fuera de la
			//diagonal (�ndice)
			int pos_rowL = diag - step;
			float k = native_divide(elmb, diag_elm);

			for (int x = tx; x < nEl-1; x += ntx) {
				int rowId = rowL[pos_rowL + x];
				b[rowId] -= k * elmL[diag + x + 1];				
			}
		}
	}
}

void Chol_sparse_5_csp(__global float* elmL,
	__global int* colL, __global float* b, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		//ubicaci�n del elemento diagonal en elmL
		int diag = colL[x];
		b[x] = native_divide(b[x], elmL[diag]);
	}
}

void Chol_sparse_6_csp(__global float* elmL,
	__global int* colL, __global int* rowL, __global float* y,
	int order, int step, __local float* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//�ndice global de hilo
	int tx = get_local_id(0);

	//n�mero total de hilos
	int ntx = get_local_size(0);

	//indice de la fila de L implicada
	int fL = order - step - 1;

	//ubicaci�n de elemento diagonal de la fila en elmL
	int diagId = colL[fL];

	//n�mero de elementos en la fila
	int nEl = (fL == order - 1 ? 1 : colL[fL + 1] - diagId);

	//ubicaci�n del �ndice del primer elemento fuera de la
	//diagonal en rowL
	int pos_rowL = diagId - fL;

	//elemento diagonal
	float diag = elmL[diagId];

	//valor donde se guardar� una suma parcial
	float sum = 0.0f;

	//solo se usar� un bloque
	if (bx == 0) {
		for (int x = tx; x < nEl-1; x += ntx) {
			int rowId = rowL[pos_rowL + x];
			sum += elmL[diagId + x + 1] * y[rowId];
		}
		partialSum[tx] = sum;

		//reducci�n en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = ntx / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			y[fL] = native_divide(y[fL] - partialSum[0], diag);
	}
}

/*Aplicado a matrices dispersas y sim�tricas almacenadas
en formato SKS (Skyline Storage):

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

//-----------------------------------
//elmL es elmA o una copia del mismo
__kernel void
chol_sparse_sks_sp(__global float* elmL,
	__global int* idxL, int size_elmL, int size_idxL,
	__global float* b, __local float* partialSum, int step,
	int sstep)
{
	//la iteraci�n principal se realiza en el host

	//se modificar� una sola columna
	if (sstep == 0)
		chol_sparse_1_sks_sp(elmL, idxL, size_idxL, step);

	//se modificar� las columnas siguientes a la columna
	//implicada
	else if (sstep == 1)
		chol_sparse_2_sks_sp(elmL, idxL, size_idxL, step);

	//una vez usada las funciones anteriores todav�a es 
	//necesario modificar la diagonal D[i]=sqrt(D[i]) para 
	//concluir la factorizaci�n
	else if (sstep == 2)
		chol_sparse_3_sks_sp(elmL, idxL, size_idxL);

	//hasta aqui la factorizaci�n esta concluida, la parte
	//triangular inferior de A contendr� la factorizaci�n
	//de Cholesky
	//desde aqui se proceder� a solucionar el sistema 
	//L*transpose(L)x=b

	//se resolver� el sistema Ly=b, y=transpose(L)*x
	else if (sstep == 3)
		chol_sparse_4_sks_sp(elmL, idxL, b, size_idxL,
			step);
	else if (sstep == 4)
		chol_sparse_5_sks_sp(elmL, idxL, b, size_idxL);

	//se resolver� el sistema transpose(L)*x=y
	//"y" ya debe haberse obtenido usando la funci�n anterior
	else
		chol_sparse_6_sks_sp(elmL, idxL, b, size_idxL,
			step, partialSum);
}

void chol_sparse_1_sks_sp(__global float* elmL,
	__global int* idxL, int order, int step)
{
	//es procedimiento solo deber� realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la �ltima 
	//columna tendr� �ndice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//obteniendo la ubicaci�n del primer elemento 
		//de la columna determinada por step
		int pos = idxL[step];

		//posici�n siguiente
		int pos_nxt = idxL[step + 1];

		//�ndice global de hilo
		int tx = get_global_id(0);

		//n�mero total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//elemento diagonal. El primer
		//elemento guardado en elmA para cada columna es el
		//elemento diagonal
		float _r = native_sqrt(elmL[pos]);

		//cada hilo se encargar� de modificar un elemento 
		//de la columna
		for (int x = tx + pos + 1; x < pos_nxt; x += ntx) {
			elmL[x] = native_divide(elmL[x], _r);
		}
	}
}

void chol_sparse_2_sks_sp(__global float* elmL,
	__global int* idxL, int order, int step)
{
	//este procedimiento solo deber� realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la �ltima 
	//columna tendr� �ndice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//�ndice local de grupo
		int bx = get_group_id(0);

		//�ndice local de hilo
		int tx = get_local_id(0);

		//posici�n del primer elemento de la columna de elmL
		int pos_elmL = idxL[step];

		//n�mero de elementos en elmL para la columna
		//sin contar la diagonal
		int nEl = idxL[step + 1] - pos_elmL - 1;

		if (nEl > 0) {
			for (int b = bx; b < nEl; b += get_num_groups(0)) {
				//indice de la columna a modificar
				int colId = step + b + 1;

				//primer factor
				float k1 = elmL[pos_elmL + b + 1];

				for (int x = b + tx; x < nEl; x += get_local_size(0)) {

					//segundo factor
					float k2 = elmL[pos_elmL + x + 1];

					//posici�n final en elmL
					int	posf = idxL[colId] + x - b;

					elmL[posf] -= k1 * k2;
				}
			}
		}
	}
}

void chol_sparse_3_sks_sp(__global float* elmL,
	__global int* idxL, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		int diag = idxL[x];
		elmL[diag] = native_sqrt(elmL[diag]);
	}
}

void chol_sparse_4_sks_sp(__global float* elmL,
	__global int* idxL, __global float* b, int order,
	int step)
{
	if (step < order - 1) {
		//�ndice global de hilo
		int tx = get_global_id(0);

		//n�mero total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//posici�n del elemento diagonal
		int diag = idxL[step];

		//n�mero de elementos para una columna
		int nEl = idxL[step + 1] - diag;

		//deben haber elementos aparte del elemento diagonal
		if (nEl > 1) {

			//elemento diagonal y elemento correspondiente en b
			float k = native_divide(b[step], elmL[diag]);

			for (int x = tx; x < nEl - 1; x += ntx) {
				int rowId = step + x + 1;
				b[rowId] -= k * elmL[diag + x + 1];
			}
		}
	}
}

void chol_sparse_5_sks_sp(__global float* elmL,
	__global int* idxL, __global float* b, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		//ubicaci�n del elemento diagonal en elmL
		int diag = idxL[x];
		b[x] = native_divide(b[x], elmL[diag]);
	}
}

void chol_sparse_6_sks_sp(__global float* elmL,
	__global int* idxL, __global float* y,
	int order, int step, __local float* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//�ndice global de hilo
	int tx = get_local_id(0);

	//n�mero total de hilos
	int ntx = get_local_size(0);

	//indice de la fila de L implicada
	int fL = order - step - 1;

	//ubicaci�n del elemento diagonal de la fila en elmL
	int diagId = idxL[fL];

	//n�mero de elementos en la fila
	int nEl = (fL == order - 1 ? 1 : idxL[fL + 1] - diagId);

	//elemento diagonal
	float diag = elmL[diagId];

	//valor donde se guardar� una suma parcial
	float sum = 0.0f;

	//solo se usar� un bloque
	if (bx == 0) {
		for (int x = tx; x < nEl - 1; x += ntx) {
			int rowId = fL + x + 1;
			sum += elmL[diagId + x + 1] * y[rowId];
		}
		partialSum[tx] = sum;

		//reducci�n en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = ntx / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			y[fL] = native_divide(y[fL] - partialSum[0],
				diag);
	}
}

//------------------------
//precisi�n doble (double)
//------------------------

/*Kernel que concibe los elementos de A, como ordenados por
columna simple precisi�n*/
__kernel void
Cholesky_c(__global double* A, __global double* b,
	__local double* partialSum, int order, int step, int sstep)
{
	//la iteraci�n principal se realiza en el host

	//se modificar� una sola columna
	if (sstep == 0) {
		Chol_1_c(A, order, step);
	}
	//se modificar� las columnas siguientes a la columna
	//implicada
	else if (sstep == 1) {
		Chol_2_c(A, order, step);
	}

	//una vez usada las funciones anteriores todav�a es 
	//necesario modificar la diagonal D[i]=sqrt(D[i]) para 
	//concluir la factorizaci�n
	else if (sstep == 2) {
		Chol_3_c(A, order);
	}

	///hasta aqui la factorizaci�n esta concluida, la parte
	//triangular inferior de A contendr� la factorizaci�n de
	//cholesky es decir L
	//desde aqui se proceder� a solucionar el sistema 
	//L*transpose(L)x=b

	//se resolver� el sistema Ly=b, y=transpose(L)*x
	else if (sstep == 3) {
		Chol_4_c(A, b, order, step);
	}
	else if (sstep == 4) {
		Chol_5_c(A, b, order);
	}

	//se resolver� el sistema transpose(L)*x=y
	//"y" ya debe haberse obtenido usando la funci�n anterior
	else {
		Chol_6_c(A, b, order, step, partialSum);
	}
}

/*
paso j(1) de la factorizaci�n de Cholesky in situ (es decir
la matriz A ser� modificada) en este paso, se modifica
la columna j.

Se modifican los elementos de la columna j a partir de la fila
j + 1.

A(n,j)=A(n,j)/sqrt(A(j,j)) n>j

se asume un almacenamiento por columna

Ver:
FACTORIZACI�N DE CHOLESKY MODIFICADA DE MATRICES DISPERSAS
SOBRE MULTIPROCESADORES
Mar�a J. Mart�n Santamar�a
Cap.2, 2.4. Factorizaci�n num�rica
2.4.1. Factorizaci�n de Cholesky est�ndar
P�g. 14
*/
void Chol_1_c(__global double* A, int order, int step)
{

	//step + 1 + �ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//columna base
	__global double* colB = A + step * order;

	//raiz cuadrada del elemento A(step,step)
	double _r = sqrt(colB[step]);

	//cada hilo se encargar� de modificar un elemento 
	//de la columna
	for (int x = tx + step + 1; x < order; x += ntx) {
		colB[x] = colB[x] / _r;
	}
}

/*
paso j(2) de la factorizaci�n de Cholesky in situ (es decir
la matriz A ser� modificada) para este paso, se modifica
las columnas siguientes a la columna j.

para una columna s, s>j, se modifican las filas n>=s

A(n,s)=A(n,s)-A(s,j)*A(n,j) s>j, n>=s

se asume un almacenamiento por columna

Ver:
FACTORIZACI�N DE CHOLESKY MODIFICADA DE MATRICES DISPERSAS
SOBRE MULTIPROCESADORES
Mar�a J. Mart�n Santamar�a
Cap.2, 2.4. Factorizaci�n num�rica
2.4.1. Factorizaci�n de Cholesky est�ndar
P�g. 14
*/
void Chol_2_c(__global double* A, int order, int step)
{
	//�ndice local de grupo
	int bx = get_group_id(0);

	//step + �ndice local de hilo
	int tx = get_local_id(0);

	//columna base
	__global double* colB = A + step * order;

	//par�metros de apoyo
	double _k;

	//en este caso, cada grupo se encargar� de las 
	//operaciones sobre todos los elementos de una 
	//columna
	for (int y = bx + step + 1; y < order; y += get_num_groups(0)) {
		_k = colB[y];
		__global double* colI = A + y * order;

		for (int x = tx + y; x < order;
			x += get_local_size(0)) {

			colI[x] -= _k * colB[x];
		}
	}
}

/*
Modifica la diagonal de A
A(j,j)=sqrt(A(j,j))

se asume un almacenamiento por columna

Ver:
FACTORIZACI�N DE CHOLESKY MODIFICADA DE MATRICES DISPERSAS
SOBRE MULTIPROCESADORES
Mar�a J. Mart�n Santamar�a
Cap.2, 2.4. Factorizaci�n num�rica
2.4.1. Factorizaci�n de Cholesky est�ndar
P�g. 14
*/
void Chol_3_c(__global double* A, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//parametros de apoyo
	int diag;

	for (int x = tx; x < order; x += ntx) {
		//ubicaci�n del elemento diagonal en memoria
		diag = x * order + x;
		A[diag] = sqrt(A[diag]);
	}
}

/*
Inicia la soluci�n del sistema Ax=b teniendo la matriz trangular
inferior L (obtenido tras la factorizaci�n de cholesky),
tal que A= L*transpose(L)

Con esta funci�n se iniciar� la soluci�n del sistema Ly=b con
y=transpose(L)*x cuya soluci�n final ser� obtenida usando
Chol_5_c

para un paso j, modificar� el vector b tal que:

b[n]=b[n]-b[j]*L[n,j]/L[j,j] n>j

este proceso es equivalente a usar el m�todo de Gauss_Jordan
en matrices triangulares inferiores, en este caso es posible
dejar intacta la matriz L modificando solo el vector b, a pesar
de que para un paso j la idea central es convertir en cero toda
esa columna, estas operaciones pueden obviarse ya que no se
necesita el valor final, por lo tanto lo que importa solo es el
trabajo que se realiza sobre el vector b

se asume un almacenamiento por columna
*/
void Chol_4_c(__global double* L, __global double* b, int order,
	int step)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//columna de L implicada
	__global double* colB = L + order * step;

	//elemeto diagonal
	double diag = colB[step];

	double k = b[step] / diag;

	for (int x = tx + step + 1; x < order; x += ntx) {
		b[x] -= k* colB[x];
	}
}

/*
divide el vector b, entre la diagonal de L
b[j]=b[j]/L[j,j]
*/
void Chol_5_c(__global double* A, __global double* b, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//parametros de apoyo
	int diag;

	for (int x = tx; x < order; x += ntx) {
		//ubicaci�n del elemento diagonal en memoria
		diag = x * order + x;
		b[x] = b[x] / A[diag];
	}
}

/*
Inicia la soluci�n del sistema Ax=b teniendo la matriz trangular
inferior L (obtenido tras la factorizaci�n de cholesky),
tal que A= L*transpose(L)

Con esta funci�n se iniciar� la soluci�n del sistema
transpose(L)*x=y, "y" debe haberse obtenido anteriormente usando
la funci�n Chol_4_csp (y= b/diag(L), al final de usar Chol_4_csp).
para un paso j modificar� el elemento b[i] tal que i=n-j:

b[i]=(b[i]-sumatoria{j=i+1,n}(y[j]*L[j,i]))/L[i,i]

este valor ser� la soluci�n final x[i] del sistema Ax=b

se asume un almacenamiento por fila, esto para poder usar la
misma matriz L sin necesidad de transponerla, adem�s se usar�
solo un bloque.
*/
void Chol_6_c(__global double* L, __global double* y, int order,
	int step, __local double* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//�ndice global de hilo
	int tx = get_local_id(0);

	//n�mero total de hilos
	int ntx = get_local_size(0);

	//indice de la fila de L implicada
	int fL = order - step - 1;

	//fila de L implicada, data
	__global double* filB = L + fL * order;

	//elemento diagonal
	double diag = filB[fL];

	//valor donde se guardar� una suma parcial
	double sum = 0.0f;

	//solo se usar� un bloque
	if (bx == 0) {
		for (int x = tx + fL + 1; x < order; x += ntx) {
			sum += filB[x] * y[x];
		}
		partialSum[tx] = sum;

		//reducci�n en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = ntx / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			y[fL] = (y[fL] - partialSum[0]) / diag;
	}
}


/*Aplicado a matrices dispersas y sim�tricas:

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


//-------------------------------
__kernel void
Cholesky_sparse_c(__global double* elmA, __global int* colA,
	__global int* rowA, __global double* elmL, __global int* colL,
	__global int* rowL, int size_elmA, int size_colA,
	__global double* b, __local double* partialSum, int step,
	int sstep)
{

	if (sstep == -1)
		sparse_fill(elmA, colA, rowA, elmL, colL,
			rowL, size_elmA);
	//la iteraci�n principal se realiza en el host

		//se modificar� una sola columna
	else if (sstep == 0)
		Chol_sparse_1(elmL, colL, size_colA, step);

	//se modificar� las columnas siguientes a la columna
	//implicada
	else if (sstep == 1)
		Chol_sparse_2(elmL, colL, rowL, size_colA, step);

	//una vez usada las funciones anteriores todav�a es 
	//necesario modificar la diagonal D[i]=sqrt(D[i]) para 
	//concluir la factorizaci�n
	else if (sstep == 2)
		Chol_sparse_3(elmL, colL, size_colA);
	//hasta aqui la factorizaci�n esta concluida, la parte
	//triangular inferior de A contendr� la factorizaci�n de
	//cholesky es decir L
	//desde aqui se proceder� a solucionar el sistema 
	//L*transpose(L)x=b

	//se resolver� el sistema Ly=b, y=transpose(L)*x
	else if (sstep == 3)
		Chol_sparse_4(elmL, colL, rowL, b, size_colA, step);
	else if (sstep == 4)
		Chol_sparse_5(elmL, colL, b, size_colA);

	//se resolver� el sistema transpose(L)*x=y
	//"y" ya debe haberse obtenido usando la funci�n anterior
	else
		Chol_sparse_6(elmL, colL, rowL, b, size_colA, step,
			partialSum);
}

/*dada dos matrices dispersas A y L, copia los elementos de
elmA en el lugar correspondiente de elmL.
Se asume que los �ndices de rowA est�n incluidos en
rowL. Es decir, en su forma densa, todos
los lugares diferentes de cero en A, tambi�n son diferentes de
cero en L, pero no viceversa*/

void sparse_fill(__global double* elmA, __global int* colA,
	__global int* rowA, __global double* elmL, __global int* colL,
	__global int* rowL, int size_elmA)
{
	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < size_elmA; x += ntx) {
		//variable donde se guardar� el �ndice de columna y de fila
		//respectivamente del elemento elmA[tx]
		int col = 0;
		int row = 0;
		//variable que controlar� la salida del bucle.
		int key = 0;

		while (key == 0) {
			if (colA[col] > x) {
				col -= 1;
				row = rowA[x - col - 1];
				key = 1;
			}
			else if (colA[col] == x) {
				row = col;
				key = 1;
			}
			else
				col += 1;
		}
		//hasta aqui para un elemento de ElmA, ya se
		//se cuenta con el �ndice de fila y de columna (row y col)
		//ahora necesitamos ubicar este elemento en elmL

		int posf;//posici�n final en elmL

		if (col != row) {
			int pos = colL[col] - col;
			int cont = 0;
			key = 0;
			while (key == 0)
				rowL[pos + cont] == row ? key = 1 : cont++;
			posf = colL[col] + cont + 1;
		}
		else
			posf = colL[col];
		//escribiendo elmentos de elmA en el lugar 
		//correspondiente de elmL
		elmL[posf] = elmA[x];
	}
}

void Chol_sparse_1(__global double* elmL, __global int* colL,
	int order, int step)
{
	//es procedimiento solo deber� realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la �ltima 
	//columna tendr� �ndice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//obteniendo la ubicaci�n del primer elemento 
		//de la columna determinada por step
		int pos = colL[step];

		//posici�n siguiente
		int pos_nxt = colL[step + 1];

		//�ndice global de hilo
		int tx = get_global_id(0);

		//n�mero total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//raiz cuadrada del elemento diagonal. El primer
		//elemento guardado en elmA para cada columna es el
		//elemento diagonal
		double _r = sqrt(elmL[pos]);

		//cada hilo se encargar� de modificar un elemento 
		//de la columna
		for (int x = tx + pos + 1; x < pos_nxt; x += ntx) {
			elmL[x] = elmL[x] / _r;
		}
	}
}

void Chol_sparse_2(__global double* elmL, __global int* colL,
	__global int* rowL, int order, int step)
{
	//es procedimiento solo deber� realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la �ltima 
	//columna tendr� �ndice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//�ndice local de grupo
		int bx = get_group_id(0);

		//�ndice local de hilo
		int tx = get_local_id(0);

		//posici�n edl primer elemento de la columna de elmL
		int pos_elmL = colL[step];

		//posici�n del primer elemento de la columna en rowL
		int pos_rowL = pos_elmL - step;

		//n�mero de elementos en rowL para la columna
		int nEl = colL[step + 1] - pos_elmL - 1;

		if (nEl > 0) {

			for (int b = bx; b < nEl; b += get_num_groups(0)) {
				//indice de la columna a modificar
				int colId = rowL[pos_rowL + b];

				//primer factor
				double k1 = elmL[pos_elmL + b + 1];

				for (int x = b + tx; x < nEl; x += get_local_size(0)) {

					//�ndice de la fila en la columna a modificar
					int rowId = rowL[pos_rowL + x];

					//segundo factor
					double k2 = elmL[pos_elmL + x + 1];

					int posf;//posici�n final en elmL

					if (colId != rowId) {
						int pos = colL[colId] - colId;
						int cont = 0;
						int key = 0;
						while (key == 0)
							rowL[pos + cont] == rowId ? key = 1 : cont++;
						posf = colL[colId] + cont + 1;
					}
					else
						posf = colL[colId];

					elmL[posf] -= k1 * k2;
				}
			}
		}
	}
}

void Chol_sparse_3(__global double* elmL,
	__global int* colL, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		int diag = colL[x];
		elmL[diag] = sqrt(elmL[diag]);
	}
}

void Chol_sparse_4(__global double* elmL,
	__global int* colL, __global int* rowL, __global double* b,
	int order, int step)
{
	if (step < order - 1) {
		//�ndice global de hilo
		int tx = get_global_id(0);

		//n�mero total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//posici�n del elemento diagonal
		int diag = colL[step];

		//n�mero de elementos para una columna
		int nEl = colL[step + 1] - diag;

		//deben haber elementos aparte del elemento diagonal
		if (nEl > 1) {

			//elemento diagonal y elemento correspondiente en b
			double diag_elm = elmL[diag];
			double elmb = b[step];

			//posici�n en rowL del primer elemento fuera de la
			//diagonal (�ndice)
			int pos_rowL = diag - step;
			double k = elmb / diag_elm;

			for (int x = tx; x < nEl - 1; x += ntx) {
				int rowId = rowL[pos_rowL + x];
				b[rowId] -= k * elmL[diag + x + 1];
			}
		}
	}
}

void Chol_sparse_5(__global double* elmL,
	__global int* colL, __global double* b, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		//ubicaci�n del elemento diagonal en elmL
		int diag = colL[x];
		b[x] = b[x]/ elmL[diag];
	}
}

void Chol_sparse_6(__global double* elmL,
	__global int* colL, __global int* rowL, __global double* y,
	int order, int step, __local double* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//�ndice global de hilo
	int tx = get_local_id(0);

	//n�mero total de hilos
	int ntx = get_local_size(0);

	//indice de la fila de L implicada
	int fL = order - step - 1;

	//ubicaci�n de elemento diagonal de la fila en elmL
	int diagId = colL[fL];

	//n�mero de elementos en la fila
	int nEl = (fL == order - 1 ? 1 : colL[fL + 1] - diagId);

	//ubicaci�n del �ndice del primer elemento fuera de la
	//diagonal en rowL
	int pos_rowL = diagId - fL;

	//elemento diagonal
	double diag = elmL[diagId];

	//valor donde se guardar� una suma parcial
	double sum = 0.0;

	//solo se usar� un bloque
	if (bx == 0) {
		for (int x = tx; x < nEl - 1; x += ntx) {
			int rowId = rowL[pos_rowL + x];
			sum += elmL[diagId + x + 1] * y[rowId];
		}
		partialSum[tx] = sum;

		//reducci�n en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = ntx / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			y[fL] = (y[fL] - partialSum[0]) / diag;
	}
}

/*Aplicado a matrices dispersas y sim�tricas almacenadas
en formato SKS (Skyline Storage):

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

//-----------------------------------
//elmL es elmA o una copia del mismo
__kernel void
chol_sparse_sks(__global double* elmL,
	__global int* idxL, int size_elmL, int size_idxL,
	__global double* b, __local double* partialSum, int step,
	int sstep)
{
	//la iteraci�n principal se realiza en el host

	//se modificar� una sola columna
	if (sstep == 0)
		chol_sparse_1_sks(elmL, idxL, size_idxL, step);

	//se modificar� las columnas siguientes a la columna
	//implicada
	else if (sstep == 1)
		chol_sparse_2_sks(elmL, idxL, size_idxL, step);

	//una vez usada las funciones anteriores todav�a es 
	//necesario modificar la diagonal D[i]=sqrt(D[i]) para 
	//concluir la factorizaci�n
	else if (sstep == 2)
		chol_sparse_3_sks(elmL, idxL, size_idxL);

	//hasta aqui la factorizaci�n esta concluida, la parte
	//triangular inferior de A contendr� la factorizaci�n
	//de Cholesky
	//desde aqui se proceder� a solucionar el sistema 
	//L*transpose(L)x=b

	//se resolver� el sistema Ly=b, y=transpose(L)*x
	else if (sstep == 3)
		chol_sparse_4_sks(elmL, idxL, b, size_idxL,
			step);
	else if (sstep == 4)
		chol_sparse_5_sks(elmL, idxL, b, size_idxL);

	//se resolver� el sistema transpose(L)*x=y
	//"y" ya debe haberse obtenido usando la funci�n anterior
	else
		chol_sparse_6_sks(elmL, idxL, b, size_idxL,
			step, partialSum);
}

void chol_sparse_1_sks(__global double* elmL,
	__global int* idxL, int order, int step)
{
	//es procedimiento solo deber� realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la �ltima 
	//columna tendr� �ndice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//obteniendo la ubicaci�n del primer elemento 
		//de la columna determinada por step
		int pos = idxL[step];

		//posici�n siguiente
		int pos_nxt = idxL[step + 1];

		//�ndice global de hilo
		int tx = get_global_id(0);

		//n�mero total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//elemento diagonal. El primer
		//elemento guardado en elmA para cada columna es el
		//elemento diagonal
		double _r = sqrt(elmL[pos]);

		//cada hilo se encargar� de modificar un elemento 
		//de la columna
		for (int x = tx + pos + 1; x < pos_nxt; x += ntx) {
			elmL[x] /= _r; 
		}
	}
}

void chol_sparse_2_sks(__global double* elmL,
	__global int* idxL, int order, int step)
{
	//este procedimiento solo deber� realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la �ltima 
	//columna tendr� �ndice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//�ndice local de grupo
		int bx = get_group_id(0);

		//�ndice local de hilo
		int tx = get_local_id(0);

		//posici�n del primer elemento de la columna de elmL
		int pos_elmL = idxL[step];

		//n�mero de elementos en elmL para la columna
		//sin contar la diagonal
		int nEl = idxL[step + 1] - pos_elmL - 1;

		if (nEl > 0) {
			for (int b = bx; b < nEl; b += get_num_groups(0)) {
				//indice de la columna a modificar
				int colId = step + b + 1;

				//primer factor
				double k1 = elmL[pos_elmL + b + 1];

				for (int x = b + tx; x < nEl; x += get_local_size(0)) {

					//segundo factor
					double k2 = elmL[pos_elmL + x + 1];

					//posici�n final en elmL
					int	posf = idxL[colId] + x - b;

					elmL[posf] -= k1 * k2;
				}
			}
		}
	}
}

void chol_sparse_3_sks(__global double* elmL,
	__global int* idxL, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		int diag = idxL[x];
		elmL[diag] = sqrt(elmL[diag]);
	}
}

void chol_sparse_4_sks(__global double* elmL,
	__global int* idxL, __global double* b, int order,
	int step)
{
	if (step < order - 1) {
		//�ndice global de hilo
		int tx = get_global_id(0);

		//n�mero total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//posici�n del elemento diagonal
		int diag = idxL[step];

		//n�mero de elementos para una columna
		int nEl = idxL[step + 1] - diag;

		//deben haber elementos aparte del elemento diagonal
		if (nEl > 1) {

			//elemento diagonal y elemento correspondiente en b
			double k = b[step] / elmL[diag];

			for (int x = tx; x < nEl - 1; x += ntx) {
				int rowId = step + x + 1;
				b[rowId] -= k * elmL[diag + x + 1];
			}
		}
	}
}

void chol_sparse_5_sks(__global double* elmL,
	__global int* idxL, __global double* b, int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		//ubicaci�n del elemento diagonal en elmL
		int diag = idxL[x];
		b[x] /= elmL[diag];
	}
}

void chol_sparse_6_sks(__global double* elmL,
	__global int* idxL, __global double* y,
	int order, int step, __local double* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//�ndice global de hilo
	int tx = get_local_id(0);

	//n�mero total de hilos
	int ntx = get_local_size(0);

	//indice de la fila de L implicada
	int fL = order - step - 1;

	//ubicaci�n del elemento diagonal de la fila en elmL
	int diagId = idxL[fL];

	//n�mero de elementos en la fila
	int nEl = (fL == order - 1 ? 1 : idxL[fL + 1] - diagId);

	//elemento diagonal
	double diag = elmL[diagId];

	//valor donde se guardar� una suma parcial
	double sum = 0.0;

	//solo se usar� un bloque
	if (bx == 0) {
		for (int x = tx; x < nEl - 1; x += ntx) {
			int rowId = fL + x + 1;
			sum += elmL[diagId + x + 1] * y[rowId];
		}
		partialSum[tx] = sum;

		//reducci�n en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = ntx / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			y[fL] = (y[fL] - partialSum[0]) / diag;
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

/*Kernel de prueba*/
__kernel void
prueba_(__global double* C, __global double* A, __global double* B,
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

/*Kernel de prueba*/
__kernel void
prueba_sp(__global float* C, __global float* A,
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