/*
=========================================
revisi�n 0.0.1 20-06-2019, 21:35 VS 2017
=========================================
*/
/* Soluci�n de un sistema de ecuaciones, usando 
el metodo de los gradientes conjugados: para un sistema 
de la forma: A*x=B donde la inc�gnita es x; A es una matriz
cuadrada sim�trica y definida positiva de orden n y B es un
vector de longitud n.

Ver:
*T�cnicas de C�lculo para Sistemas de Ecuaciones,
Programaci�n Lineal y Programaci�n Entera
Jos� Luis de la Fuente O�Connor
Cap.2, 2.5. M�todos de minimizaci�n
2.5.2 Direcciones de descenso conjugadas
2.5.2.2 Determinaci�n de direcciones conjugadas. M�todo de
los gradientes conjugados
P�g. 179

 */

void __gpu_sync(int goalVal,
	__global volatile int * Arrayin, __global volatile int * Arrayout);

void __gpu_sync1(int goalVal,
	__global volatile int * g_mutex);

//float
float atom_add_float(__global float* address, float val);
void sparse_mat_vec_mul1_sp(__global float* elmA,
	__global int* colA, __global int* rowA, __global float* b,
	__global float* c, int order, __local float* partialSum);
void sparse_mat_vec_mul2_sp(__global float* elmA,
	__global int* colA, __global int* rowA, __global float* b,
	__global float* c, int order);
void sparse_mat_vec_mul3_sp(__global float* elmA,
	__global int* colA, __global float* b, __global float* c,
	int order);

//double
double atom_add_double(__global double* address, double val);
void sparse_mat_vec_mul1(__global double* elmA,
	__global int* colA, __global int* rowA, __global double* b,
	__global double* c, int order, __local double* partialSum);
void sparse_mat_vec_mul2(__global double* elmA,
	__global int* colA, __global int* rowA, __global double* b,
	__global double* c, int order);
void sparse_mat_vec_mul3(__global double* elmA,
	__global int* colA, __global double* b, __global double* c,
	int order);
//------------------------
//simple precisi�n (float)
//------------------------

/*Kernel que concibe los elementos de A, como ordenados por 
columna simple precisi�n*/
__kernel void
gconj_c_sp(__global float* elmA, __global int* colA, 
	__global int* rowA, __global float* b,__global float* c,
	__local float* partialSum, int order, int step, int sstep)
{
	/*//la iteraci�n principal se realiza en el host

	//multiplicaci�n de matriz por vector
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
	}*/
}

__kernel void
sparse_mat_vec_mul_sp(__global float* elmA, __global int* colA,
	__global int* rowA, __global float* b, __global float* c,
	__local float* partialSum, int order, int step)
{
	//multiplicaci�n de matriz por vector

	//transpose(L)*b, los elementos de c ser�n reemplazados
	//c[i]=valor
	if (step == 0)
		sparse_mat_vec_mul1_sp(elmA, colA, rowA, b, c, order,
			partialSum);
	//L*b, los elementos de c se acumular�n apartir de su
	//valor actual 
	//c[i]+=valor o valores
	else if (step == 1)
		sparse_mat_vec_mul2_sp(elmA, colA, rowA, b, c, order);

	//D*b, los elementos de c se acumular�n a partir de su
	//valor actual
	//c[i]+=valor
	else if (step == 2)
		sparse_mat_vec_mul3_sp(elmA, colA, b, c, order);
	else;
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
*/

/*la multiplicaci�n de una matriz dispersa y sim�trica 
guardada en el formato establecido, se realizar� en tres
partes. Ya que el almacenamiento se hace solo de los 
elementos distintos de cero de la parte triangular superior
o inferior (que en este caso es el mismo), la matriz en
su forma densa puede descomponerse en tres t�rminos, una matriz
triangular inferior,una matriz triangular superior y una matriz
diagonal, es decir:

sea A una matriz cuadrada y sim�trica de orden n

L una matriz triangular inferior con los elementos iguales
a la parte triangular inferior de A, excepto la diagonal 
(estos ser�n iguales a cero).

U una matriz triangular superior con los elementos iguales
a la parte triangular superior de A, excepto la diagonal 
(estos ser�n iguales a cero).

D una matriz diagonal con los elementos iguales a la diagonal
de A.

entonces se cumple que L+D+U=A

ya que se trata de una matriz sim�trica entonces
U=transpose(L) por lo que:

L+D+transpose(L)=A

al multiplciar por un vector b puede descomponerse en:
A*b=L*b + D*b + transpose(L)*b

para multiplicar transpose(L)*b puede usarse L sin modificar
ninguno de sus campos, solo debe asumirse el almacenamiento 
por filas en vez de por columnas para tal caso

*/

//transpose(L)*b
void sparse_mat_vec_mul1_sp(__global float* elmA,
	__global int* colA, __global int* rowA, __global float* b,
	__global float* c, int order, __local float* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//�ndice local de hilo
	int tx = get_local_id(0);

	for (int row = bx; row < order; row += get_num_groups(0)) {

		//ubicaci�n de elemento diagonal de la fila en elmL
		int dId = colA[row];

		//n�mero de elementos en la fila
		int nEl = (row == order - 1 ? 1 : colA[row + 1] - dId);

		//ubicaci�n del �ndice del primer elemento fuera de la
		//diagonal en rowL
		int baseId = dId - row;

		//valor donde se guardar� una suma parcial
		float sum = 0.0f;

		for (int x = tx; x < nEl - 1; x += get_local_size(0)) {
			int cId = rowA[baseId + x];
			sum += elmA[dId + x + 1] * b[cId];
		}
		partialSum[tx] = sum;

		//reducci�n en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			c[row] = partialSum[0];
	}
}

//L*b
void sparse_mat_vec_mul2_sp(__global float* elmA,
	__global int* colA, __global int* rowA, __global float* b,
	__global float* c, int order)
{
	//�ndice de grupo
	int bx = get_group_id(0);

	//�ndice local de hilo
	int tx = get_local_id(0);

	for (int col = bx; col < order; col += get_num_groups(0)) {
		//posici�n del elemento diagonal
		int dId = colA[col];

		//n�mero de elementos para una columna
		int nEl = (col == order - 1 ? 1 : colA[col + 1] - dId);

		//posici�n en rowL del primer elemento fuera de la
		//diagonal (�ndice)
		int baseId = dId - col;

		//elemento en b, por el que se multiplica
		float mul = b[col];

		for (int x = tx; x < nEl - 1; x += get_local_size(0)) {
			int rId = rowA[baseId + x];
			float sum = mul * elmA[dId + x + 1];
			atom_add_float(&(c[rId]), sum);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//D*b
void sparse_mat_vec_mul3_sp(__global float* elmA,
	__global int* colA, __global float* b,__global float* c,
	int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		//ubicaci�n del elemento diagonal en elmL
		int diag = colA[x];
		c[x] += b[x] * elmA[diag];
	}
}

//------------------------
//precisi�n doble (double)
//------------------------

/*Kernel que concibe los elementos de A, como ordenados por
columna simple precisi�n*/
__kernel void
gconj_c(__global double* elmA, __global int* colA,
	__global int* rowA, __global double* b, __global double* c,
	__local double* partialSum, int order, int step, int sstep)
{
	/*//la iteraci�n principal se realiza en el host

	//multiplicaci�n de matriz por vector
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
	}*/
}

__kernel void
sparse_mat_vec_mul(__global double* elmA, __global int* colA,
	__global int* rowA, __global double* b, __global double* c,
	__local double* partialSum, int order, int step)
{
	//multiplicaci�n de matriz por vector

	//transpose(L)*b, los elementos de c ser�n reemplazados
	//c[i]=valor
	if (step == 0)
		sparse_mat_vec_mul1(elmA, colA, rowA, b, c, order,
			partialSum);
	//L*b, los elementos de c se acumular�n apartir de su
	//valor actual 
	//c[i]+=valor o valores
	else if (step == 1)
		sparse_mat_vec_mul2(elmA, colA, rowA, b, c, order);

	//D*b, los elementos de c se acumular�n a partir de su
	//valor actual
	//c[i]+=valor
	else if (step == 2)
		sparse_mat_vec_mul3(elmA, colA, b, c, order);
	else;
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
*/

/*la multiplicaci�n de una matriz dispersa y sim�trica
guardada en el formato establecido, se realizar� en tres
partes. Ya que el almacenamiento se hace solo de los
elementos distintos de cero de la parte triangular superior
o inferior (que en este caso es el mismo), la matriz en
su forma densa puede descomponerse en tres t�rminos, una matriz
triangular inferior,una matriz triangular superior y una matriz
diagonal, es decir:

sea A una matriz cuadrada y sim�trica de orden n

L una matriz triangular inferior con los elementos iguales
a la parte triangular inferior de A, excepto la diagonal
(estos ser�n iguales a cero).

U una matriz triangular superior con los elementos iguales
a la parte triangular superior de A, excepto la diagonal
(estos ser�n iguales a cero).

D una matriz diagonal con los elementos iguales a la diagonal
de A.

entonces se cumple que L+D+U=A

ya que se trata de una matriz sim�trica entonces
U=transpose(L) por lo que:

L+D+transpose(L)=A

al multiplciar por un vector b puede descomponerse en:
A*b=L*b + D*b + transpose(L)*b

para multiplicar transpose(L)*b puede usarse L sin modificar
ninguno de sus campos, solo debe asumirse el almacenamiento
por filas en vez de por columnas para tal caso

*/

//transpose(L)*b
void sparse_mat_vec_mul1(__global double* elmA,
	__global int* colA, __global int* rowA, __global double* b,
	__global double* c, int order, __local double* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//�ndice local de hilo
	int tx = get_local_id(0);

	for (int row = bx; row < order; row += get_num_groups(0)) {

		//ubicaci�n de elemento diagonal de la fila en elmL
		int dId = colA[row];

		//n�mero de elementos en la fila
		int nEl = (row == order - 1 ? 1 : colA[row + 1] - dId);

		//ubicaci�n del �ndice del primer elemento fuera de la
		//diagonal en rowL
		int baseId = dId - row;

		//valor donde se guardar� una suma parcial
		double sum = 0.0f;

		for (int x = tx; x < nEl - 1; x += get_local_size(0)) {
			int cId = rowA[baseId + x];
			sum += elmA[dId + x + 1] * b[cId];
		}
		partialSum[tx] = sum;

		//reducci�n en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			c[row] = partialSum[0];
	}
}

//L*b
void sparse_mat_vec_mul2(__global double* elmA,
	__global int* colA, __global int* rowA, __global double* b,
	__global double* c, int order)
{
	//�ndice de grupo
	int bx = get_group_id(0);

	//�ndice local de hilo
	int tx = get_local_id(0);

	for (int col = bx; col < order; col += get_num_groups(0)) {
		//posici�n del elemento diagonal
		int dId = colA[col];

		//n�mero de elementos para una columna
		int nEl = (col == order - 1 ? 1 : colA[col + 1] - dId);

		//posici�n en rowL del primer elemento fuera de la
		//diagonal (�ndice)
		int baseId = dId - col;

		//elemento en b, por el que se multiplica
		double mul = b[col];

		for (int x = tx; x < nEl - 1; x += get_local_size(0)) {
			int rId = rowA[baseId + x];
			double sum = mul * elmA[dId + x + 1];
			atom_add_double(&(c[rId]), sum);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//D*b
void sparse_mat_vec_mul3(__global double* elmA,
	__global int* colA, __global double* b, __global double* c,
	int order)
{

	//�ndice global de hilo
	int tx = get_global_id(0);

	//n�mero total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		//ubicaci�n del elemento diagonal en elmL
		int diag = colA[x];
		c[x] += b[x] * elmA[diag];
	}
}

//funci�n atom_add para valores del tipo float en memoria
//global
float atom_add_float(__global float* address, float val) {
	__global int* address_as_ull =
		(__global int*)address;
	int old = *address_as_ull;
	int assumed;

	do {
		assumed = old;
		old = atom_cmpxchg(address_as_ull, assumed,
			as_int(val + as_float(assumed)));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return as_float(old);
}

//funci�n atom_add para valores del tipo float en memoria
//global
double atom_add_double(__global double* address, double val) {
	__global long* address_as_ull =
		(__global long*)address;
	long old = *address_as_ull;
	long assumed;

	do {
		assumed = old;
		old = atom_cmpxchg(address_as_ull, assumed,
			as_long(val + as_double(assumed)));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return as_double(old);
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
	barrier(CLK_GLOBAL_MEM_FENCE);
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
	barrier(CLK_GLOBAL_MEM_FENCE);
}

/*Kernel de prueba*/
__kernel void
prueba_1_sp(__global volatile float* C,__global int* D, int order)
{
	int tx = get_local_id(0);
	int bx = get_group_id(0);
	
	for (int col = bx; col < order; col += get_num_groups(0)) {
		for (int x = tx; x < order; x += get_local_size(0)) {

			atom_add_float(&(C[col]), x);
			atom_add(&(D[col]), x);

		}
	}
}

__kernel void
prueba_1(__global double* C, __global int* D, int order)
{
	int tx = get_local_id(0);
	int bx = get_group_id(0);

	for (int col = bx; col < order; col += get_num_groups(0)) {
		for (int x = tx; x < order; x += get_local_size(0)) {

			atom_add_double(&(C[col]), x);
			atom_add(&(D[col]), x);

		}
	}
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