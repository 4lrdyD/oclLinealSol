/*
=========================================
revisión 0.0.5 14-01-2020, 00:50 VS 2017
=========================================
*/
/* Solución de un sistema de ecuaciones, usando 
el metodo de los gradientes conjugados: para un sistema 
de la forma: A*x=B donde la incógnita es x; A es una matriz
cuadrada simétrica y definida positiva de orden n y B es un
vector de longitud n.

Ver:
*Técnicas de Cálculo para Sistemas de Ecuaciones,
Programación Lineal y Programación Entera
José Luis de la Fuente O’Connor
Cap.2, 2.5. Métodos de minimización
2.5.2 Direcciones de descenso conjugadas
2.5.2.2 Determinación de direcciones conjugadas. Método de
los gradientes conjugados
Pág. 179

 */

void __gpu_sync(int goalVal,
	__global volatile int * Arrayin, __global volatile int * Arrayout);

void __gpu_sync1(int goalVal,
	__global volatile int * g_mutex);

//float
float atom_add_float(__global float* address, float val);
void mat_vec_mul_sp(__global float* A, __global float* b,
	__local float* partialSum, __global float* result,
	int order);
void vec_vec_mul_sp(__global float* a, __global float* b,
	__local float* partialSum, __global float* result,
	int size);
void vec_vec_mul1_sp(__global float* a, __global float* b,
	__local float* partialSum, __global float* result,
	int size);
void sparse_mat_vec_mul1_sp(__global float* elmA,
	__global int* colA, __global int* rowA, __global float* b,
	__global float* c, int order, __local float* partialSum);
void sparse_mat_vec_mul2_sp(__global float* elmA,
	__global int* colA, __global int* rowA, __global float* b,
	__global float* c, int order);
void sparse_mat_vec_mul3_sp(__global float* elmA,
	__global int* colA, __global float* b, __global float* c,
	int order);

void sparse_mat_vec_mul1_sks_sp(__global float* elmA,
	__global int* idxA, __global float* b,
	__global float* c, int order,
	__local float* partialSum);
void sparse_mat_vec_mul2_sks_sp(__global float* elmA,
	__global int* idxA, __global float* b,
	__global float* c, int order);

//double
double atom_add_double(__global double* address, double val);
void mat_vec_mul(__global double* A, __global double* b,
	__local double* partialSum, __global double* result,
	int order);
void mat_vec_mul_sub(__global double* A, __global double* x0,
	__global double* b, __local double* partialSum,
	__global double* r, int order);
void vec_vec_mul(__global double* a, __global double* b,
	__local double* partialSum, __global double* result,
	int size);
void vec_vec_mul1(__global double* a, __global double* b,
	__local double* partialSum, __global double* result,
	int size);
void norm_euclid_pow2(__global double* v,
	__global double* norm, __local double* partialSum,
	int size);
void sparse_mat_vec_mul1(__global double* elmA,
	__global int* colA, __global int* rowA, __global double* b,
	__global double* c, int order, __local double* partialSum);
void sparse_mat_vec_mul2(__global double* elmA,
	__global int* colA, __global int* rowA, __global double* b,
	__global double* c, int order);
void sparse_mat_vec_mul3(__global double* elmA,
	__global int* colA, __global double* b, __global double* c,
	int order);

void sparse_mat_vec_mul1_sks(__global double* elmA,
	__global int* idxA, __global double* b,
	__global double* c, int order,
	__local double* partialSum);
void sparse_mat_vec_mul2_sks(__global double* elmA,
	__global int* idxA, __global double* b,
	__global double* c, int order);

//------------------------
//simple precisión (float)
//------------------------

/*Kernel que concibe los elementos de A, como ordenados por 
columna simple precisión*/
__kernel void
gconj_c_sp(__global float* a, __global float* b,
	__local float* partialSum, __global float* result,
	int size)
{
	mat_vec_mul_sp(a, b, partialSum, result, size);
}

void mat_vec_mul_sp(__global float* A, __global float* b,
	__local float* partialSum, __global float* result,
	int order) {
	//índices de hilo (local) y de grupo
	int tx = get_local_id(0);
	int bx = get_group_id(0);

	for (int y = bx; y < order; y+= get_num_groups(0)) {
		float sum = 0;
		//fila de A, determinada por y
		__global float* row = A + y * order;
		for (int x = tx; x < order; x += get_local_size(0))
			sum += row[x] * b[x];
		partialSum[tx] = sum;
		//suma por reducción en paralelo
		for (int stride = get_local_size(0) / 2;
			stride > 0; stride /= 2) {
			//sincronizando
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride)
				partialSum[tx] += partialSum[tx + stride];
		}
		//solo un hilo realiza la escritura
		if (tx == 0)
			result[y] = partialSum[0];
	}
}

void vec_vec_mul_sp(__global float* a, __global float* b,
	__local float* partialSum, __global float* result,
	int size) {
	//índices de hilo (local) y de grupo
	int tx = get_local_id(0);
	int bx = get_group_id(0);
	//solo se usará un grupo
	if (bx == 0) {
		float sum = 0;
		for (int x = tx; x < size; x+=get_local_size(0))
			sum += a[x] * b[x];
		partialSum[tx] = sum;
		//suma por reducción en paralelo
		for (int stride = get_local_size(0) / 2;
			stride > 0; stride /= 2) {
			//sincronizando
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride)
				partialSum[tx] += partialSum[tx + stride];
		}
		//solo un hilo realiz la escritura
		if (tx == 0)
			result[0] = partialSum[0];
	}
}

void vec_vec_mul1_sp(__global float* a, __global float* b,
	__local float* partialSum, __global float* result,
	int size) {
	//índices de hilo (local y global) y de grupo
	int tx = get_local_id(0);
	int tgx = get_global_id(0);
	int bx = get_group_id(0);
	//para almacenar una suma parcial
	float sum = 0;
	for (int x = tgx; x < size;	x += get_global_size(0)) {
		sum += a[x] * b[x];
	}
	partialSum[tx] = sum;
	//suma por reducción en paralelo
	for (int stride = get_local_size(0) / 2;
		stride > 0; stride /= 2) {
		//sincronizando
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tx < stride)
			partialSum[tx] += partialSum[tx + stride];
	}
	//solo un hilo realiz la escritura
	if (tx == 0)
		//ya que varios bloques tienen una suma parcial
		//necesitamos acumular usando la función atómica
		atom_add_float(&(result[0]), partialSum[0]);
}

/*Aplicado a matrices dispersas y simétricas:

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
*/
/*la multiplicación de una matriz dispersa y simétrica
guardada en el formato establecido, se realizará en tres
partes. Ya que el almacenamiento se hace solo de los
elementos distintos de cero de la parte triangular superior
o inferior (que en este caso es el mismo), la matriz en
su forma densa puede descomponerse en tres términos, una matriz
triangular inferior,una matriz triangular superior y una matriz
diagonal, es decir:

sea A una matriz cuadrada y simétrica de orden n

L una matriz triangular inferior con los elementos iguales
a la parte triangular inferior de A, excepto la diagonal
(estos serán iguales a cero).

U una matriz triangular superior con los elementos iguales
a la parte triangular superior de A, excepto la diagonal
(estos serán iguales a cero).

D una matriz diagonal con los elementos iguales a la diagonal
de A.

entonces se cumple que L+D+U=A

ya que se trata de una matriz simétrica entonces
U=transpose(L) por lo que:

L+D+transpose(L)=A

al multiplciar por un vector b puede descomponerse en:
A*b=L*b + D*b + transpose(L)*b

para multiplicar transpose(L)*b puede usarse L sin modificar
ninguno de sus campos, solo debe asumirse el almacenamiento
por filas en vez de por columnas para tal caso

*/
__kernel void
sparse_mat_vec_mul_sp(__global float* elmA, __global int* colA,
	__global int* rowA, __global float* b, __global float* c,
	__local float* partialSum, int order, int step)
{
	//multiplicación de matriz por vector

	//transpose(L)*b, los elementos de c serán reemplazados
	//c[i]=valor
	if (step == 0)
		sparse_mat_vec_mul1_sp(elmA, colA, rowA, b, c, order,
			partialSum);
	//L*b, los elementos de c se acumularán apartir de su
	//valor actual 
	//c[i]+=valor o valores
	else if (step == 1)
		sparse_mat_vec_mul2_sp(elmA, colA, rowA, b, c, order);

	//D*b, los elementos de c se acumularán a partir de su
	//valor actual
	//c[i]+=valor
	else if (step == 2)
		sparse_mat_vec_mul3_sp(elmA, colA, b, c, order);
	else;
}

//transpose(L)*b
void sparse_mat_vec_mul1_sp(__global float* elmA,
	__global int* colA, __global int* rowA, __global float* b,
	__global float* c, int order, __local float* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//índice local de hilo
	int tx = get_local_id(0);

	for (int row = bx; row < order; row += get_num_groups(0)) {

		//ubicación de elemento diagonal de la fila en elmL
		int dId = colA[row];

		//número de elementos en la fila
		int nEl = (row == order - 1 ? 1 : colA[row + 1] - dId);

		//ubicación del índice del primer elemento fuera de la
		//diagonal en rowL
		int baseId = dId - row;

		//valor donde se guardará una suma parcial
		float sum = 0.0f;

		for (int x = tx; x < nEl - 1; x += get_local_size(0)) {
			int cId = rowA[baseId + x];
			sum += elmA[dId + x + 1] * b[cId];
		}
		partialSum[tx] = sum;

		//reducción en paralelo pare determinar la suma de
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
	//índice de grupo
	int bx = get_group_id(0);

	//índice local de hilo
	int tx = get_local_id(0);

	for (int col = bx; col < order; col += get_num_groups(0)) {
		//posición del elemento diagonal
		int dId = colA[col];

		//número de elementos para una columna
		int nEl = (col == order - 1 ? 1 : colA[col + 1] - dId);

		//posición en rowL del primer elemento fuera de la
		//diagonal (índice)
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

	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		//ubicación del elemento diagonal en elmL
		int diag = colA[x];
		c[x] += b[x] * elmA[diag];
	}
}

//Multiplicación matriz(SKS)-vector
//---------------------------------
/*Aplicado a matrices dispersas y simétricas:

Sea una matriz representada en su forma densa por A.
esta matriz en su forma dispersa almacenada en formato
SKS (Skyline Storage) será representada
por 2 vectores.

*elmA: Vector de elementos distintos de cero, guardados
columna a columna.

*idxA: Vector de índices, donde cada elemento indica la
ubicación en elmA del primer elemento distinto de cero en
cada columna de A

para más detalle sobre el formato de almacenamiento SKS 
ver ayuda de los kernels en los archivos cl de la 
factorización de Cholesky
*/

/*la multiplicación de una matriz dispersa y simétrica
guardada en el formato SKS, se realizará en dos
partes. Ya que el almacenamiento se hace solo de los
elementos distintos de cero de la parte triangular superior
o inferior (que en este caso es lo mismo), la matriz en
su forma densa puede descomponerse en dos términos,
una matriz triangular inferior con diagonal cero y una
matriz triangular superior, es decir:

sea A una matriz cuadrada y simétrica de orden n

L una matriz triangular inferior con los elementos iguales
a la parte triangular inferior de A, excepto la diagonal
(estos serán iguales a cero).

U una matriz triangular superior con los elementos iguales
a la parte triangular superior de A, incluyendo 
la diagonal.

entonces se cumple que U+L=A

al multiplciar por un vector b puede descomponerse en:
A*b=U*b + L*b

para multiplicar U*b puede usarse L sin modificar
ninguno de sus campos, solo debe asumirse el almacenamiento
por filas en vez de por columnas para tal caso
*/
__kernel void
sparse_mat_vec_mul_sks_sp(__global float* elmA,
	__global int* idxA, __global float* b,
	__global float* c, __local float* partialSum,
	int order, int step)
{
	//multiplicación de matriz por vector

	//U*b, los elementos de c serán reemplazados
	//c[i]=valor
	if (step == 0)
		sparse_mat_vec_mul1_sks_sp(elmA, idxA, b, c,
			order, partialSum);
	//L*b, los elementos de c se acumularán apartir de su
	//valor actual 
	//c[i]+=valor o valores
	else if (step == 1)
		sparse_mat_vec_mul2_sks_sp(elmA, idxA, b, c,
			order);
	else;
}

//U*b
void sparse_mat_vec_mul1_sks_sp(__global float* elmA,
	__global int* idxA, __global float* b,
	__global float* c, int order,
	__local float* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//índice local de hilo
	int tx = get_local_id(0);

	for (int row = bx; row < order; row += get_num_groups(0)) {

		//ubicación de elemento diagonal de la fila en elmA
		int dId = idxA[row];

		//número de elementos en la fila
		int nEl = (row == order - 1 ? 1 : idxA[row + 1] - dId);

		//valor donde se guardará una suma parcial
		float sum = 0.0f;

		for (int x = tx; x < nEl; x += get_local_size(0)) {
			int cId = row + x;
			sum += elmA[dId + x] * b[cId];
		}
		partialSum[tx] = sum;

		//reducción en paralelo pare determinar la suma de
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
void sparse_mat_vec_mul2_sks_sp(__global float* elmA,
	__global int* idxA, __global float* b,
	__global float* c, int order)
{
	//índice de grupo
	int bx = get_group_id(0);

	//índice local de hilo
	int tx = get_local_id(0);

	for (int col = bx; col < order; col += get_num_groups(0)) {
		//posición del elemento diagonal
		int dId = idxA[col];

		//número de elementos para una columna
		int nEl = (col == order - 1 ? 1 : idxA[col + 1] - dId);

		//posición en rowL del primer elemento fuera de la
		//diagonal (índice)
		int baseId = dId - col;

		//elemento en b, por el que se multiplica
		float mul = b[col];

		for (int x = tx; x < nEl - 1; x += get_local_size(0)) {
			int rId = col + x + 1;
			float sum = mul * elmA[dId + x + 1];
			atom_add_float(&(c[rId]), sum);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//------------------------
//precisión doble (double)
//------------------------

/*Kernel que concibe los elementos de A, como ordenados por
columna simple precisión*/
__kernel void
gconj_c(__global double* A, __global double* b,
	__global double* r, __global double* a,
	__global double* z,__global double* p,
	__local double* partialSum,
	__global double* norm_pow2,
	__global int* key,
	int order)
{
	int tgx = get_global_id(0);
	//calculará r-=a.*z y el cuadrado de la norma de r
	if (key[0] == 0) {
		double val = a[0]; 
		for (int x = tgx; x < order;
			x += get_global_size(0)) 
			r[x] -= val * z[x];
		norm_euclid_pow2(r, norm_pow2, partialSum, order);
		//sumando uno a key
		if (tgx == 0)
			key[0]++;
	}

	//else if

	//mat_vec_mul_sub(A, b, b, partialSum, a, order);
}

//result=Ab (A:matriz nxn, b:vector n)
void mat_vec_mul(__global double* A, __global double* b,
	__local double* partialSum, __global double* result,
	int order) {
	//índices de hilo (local) y de grupo
	int tx = get_local_id(0);
	int bx = get_group_id(0);

	for (int y = bx; y < order; y += get_num_groups(0)) {
		double sum = 0;
		//fila de A, determinada por y
		__global double* row = A + y * order;
		for (int x = tx; x < order; x += get_local_size(0))
			sum += row[x] * b[x];
		partialSum[tx] = sum;
		//suma por reducción en paralelo
		for (int stride = get_local_size(0) / 2;
			stride > 0; stride /= 2) {
			//sincronizando
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride)
				partialSum[tx] += partialSum[tx + stride];
		}
		//solo un hilo realiza la escritura
		if (tx == 0)
			result[y] = partialSum[0];
	}
}

//r=b-Ax0 (A:matriz nxn, b,x0:vectores n)
void mat_vec_mul_sub(__global double* A, __global double* x0,
	__global double* b,__local double* partialSum,
	__global double* r, int order) {
	//índices de hilo (local) y de grupo
	int tx = get_local_id(0);
	int bx = get_group_id(0);

	for (int y = bx; y < order; y += get_num_groups(0)) {
		double sum = 0;
		//fila de A, determinada por y
		__global double* row = A + y * order;
		for (int x = tx; x < order; x += get_local_size(0))
			sum += row[x] * x0[x];
		partialSum[tx] = sum;
		//suma por reducción en paralelo
		for (int stride = get_local_size(0) / 2;
			stride > 0; stride /= 2) {
			//sincronizando
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride)
				partialSum[tx] += partialSum[tx + stride];
		}
		//solo un hilo realiza la escritura
		if (tx == 0)
			r[y] = b[y] - partialSum[0];
	}
}

void vec_vec_mul(__global double* a, __global double* b,
	__local double* partialSum, __global double* result,
	int size) {
	//índices de hilo (local) y de grupo
	int tx = get_local_id(0);
	int bx = get_group_id(0);
	//solo se usará un grupo
	if (bx == 0) {
		double sum = 0;
		for (int x = tx; x < size; x += get_local_size(0))
			sum += a[x] * b[x];
		partialSum[tx] = sum;
		//suma por reducción en paralelo
		for (int stride = get_local_size(0) / 2;
			stride > 0; stride /= 2) {
			//sincronizando
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride)
				partialSum[tx] += partialSum[tx + stride];
		}
		//solo un hilo realiz la escritura
		if (tx == 0)
			result[0] = partialSum[0];
	}
}

void vec_vec_mul1(__global double* a, __global double* b,
	__local double* partialSum, __global double* result,
	int size) {
	//índices de hilo (local y global) y de grupo
	int tx = get_local_id(0);
	int tgx = get_global_id(0);
	int bx = get_group_id(0);
	//para almacenar una suma parcial
	double sum = 0;
	for (int x = tgx; x < size; x += get_global_size(0)) {
		sum += a[x] * b[x];
	}
	partialSum[tx] = sum;
	//suma por reducción en paralelo
	for (int stride = get_local_size(0) / 2;
		stride > 0; stride /= 2) {
		//sincronizando
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tx < stride)
			partialSum[tx] += partialSum[tx + stride];
	}
	//solo un hilo realiz la escritura
	if (tx == 0)
		//ya que varios bloques tienen una suma parcial
		//necesitamos acumular usando la función atómica
		atom_add_float(&(result[0]), partialSum[0]);
}

//calculará el cuadrado de la norma euclidea de v
void norm_euclid_pow2(__global double* v,
	__global double* norm,__local double* partialSum,
	int size) {
	//índices de hilo (local y global) y de grupo
	int tx = get_local_id(0);
	int tgx = get_global_id(0);
	int bx = get_group_id(0);
	//para almacenar una suma parcial
	double sum = 0;
	for (int x = tgx; x < size; x += get_global_size(0)) {
		sum += pown(v[x], 2);
	}
	partialSum[tx] = sum;
	//suma por reducción en paralelo
	for (int stride = get_local_size(0) / 2;
		stride > 0; stride /= 2) {
		//sincronizando
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tx < stride)
			partialSum[tx] += partialSum[tx + stride];
	}
	//solo un hilo realiz la escritura
	if (tx == 0) 
		//ya que varios bloques tienen una suma parcial
		//necesitamos acumular usando la función atómica
		atom_add_double(&(norm[0]), partialSum[0]);
}


/*Aplicado a matrices dispersas y simétricas:

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
*/

/*la multiplicación de una matriz dispersa y simétrica
guardada en el formato establecido, se realizará en tres
partes. Ya que el almacenamiento se hace solo de los
elementos distintos de cero de la parte triangular superior
o inferior (que en este caso es el mismo), la matriz en
su forma densa puede descomponerse en tres términos, una matriz
triangular inferior,una matriz triangular superior y una matriz
diagonal, es decir:

sea A una matriz cuadrada y simétrica de orden n

L una matriz triangular inferior con los elementos iguales
a la parte triangular inferior de A, excepto la diagonal
(estos serán iguales a cero).

U una matriz triangular superior con los elementos iguales
a la parte triangular superior de A, excepto la diagonal
(estos serán iguales a cero).

D una matriz diagonal con los elementos iguales a la diagonal
de A.

entonces se cumple que L+D+U=A

ya que se trata de una matriz simétrica entonces
U=transpose(L) por lo que:

L+D+transpose(L)=A

al multiplciar por un vector b puede descomponerse en:
A*b=L*b + D*b + transpose(L)*b

para multiplicar transpose(L)*b puede usarse L sin modificar
ninguno de sus campos, solo debe asumirse el almacenamiento
por filas en vez de por columnas para tal caso

*/
__kernel void
sparse_mat_vec_mul(__global double* elmA, __global int* colA,
	__global int* rowA, __global double* b, __global double* c,
	__local double* partialSum, int order, int step)
{
	//multiplicación de matriz por vector

	//transpose(L)*b, los elementos de c serán reemplazados
	//c[i]=valor
	if (step == 0)
		sparse_mat_vec_mul1(elmA, colA, rowA, b, c, order,
			partialSum);
	//L*b, los elementos de c se acumularán apartir de su
	//valor actual 
	//c[i]+=valor o valores
	else if (step == 1)
		sparse_mat_vec_mul2(elmA, colA, rowA, b, c, order);

	//D*b, los elementos de c se acumularán a partir de su
	//valor actual
	//c[i]+=valor
	else if (step == 2)
		sparse_mat_vec_mul3(elmA, colA, b, c, order);
	else;
}

//transpose(L)*b
void sparse_mat_vec_mul1(__global double* elmA,
	__global int* colA, __global int* rowA, __global double* b,
	__global double* c, int order, __local double* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//índice local de hilo
	int tx = get_local_id(0);

	for (int row = bx; row < order; row += get_num_groups(0)) {

		//ubicación de elemento diagonal de la fila en elmL
		int dId = colA[row];

		//número de elementos en la fila
		int nEl = (row == order - 1 ? 1 : colA[row + 1] - dId);

		//ubicación del índice del primer elemento fuera de la
		//diagonal en rowL
		int baseId = dId - row;

		//valor donde se guardará una suma parcial
		double sum = 0.0;

		for (int x = tx; x < nEl - 1; x += get_local_size(0)) {
			int cId = rowA[baseId + x];
			sum += elmA[dId + x + 1] * b[cId];
		}
		partialSum[tx] = sum;

		//reducción en paralelo pare determinar la suma de
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
	//índice de grupo
	int bx = get_group_id(0);

	//índice local de hilo
	int tx = get_local_id(0);

	for (int col = bx; col < order; col += get_num_groups(0)) {
		//posición del elemento diagonal
		int dId = colA[col];

		//número de elementos para una columna
		int nEl = (col == order - 1 ? 1 : colA[col + 1] - dId);

		//posición en rowL del primer elemento fuera de la
		//diagonal (índice)
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

	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		//ubicación del elemento diagonal en elmL
		int diag = colA[x];
		c[x] += b[x] * elmA[diag];
	}
}

//Multiplicación matriz(SKS)-vector
//---------------------------------
__kernel void
sparse_mat_vec_mul_sks(__global double* elmA,
	__global int* idxA, __global double* b,
	__global double* c, __local double* partialSum,
	int order, int step)
{
	//multiplicación de matriz por vector

	//U*b, los elementos de c serán reemplazados
	//c[i]=valor
	if (step == 0)
		sparse_mat_vec_mul1_sks(elmA, idxA, b, c,
			order, partialSum);
	//L*b, los elementos de c se acumularán apartir de su
	//valor actual 
	//c[i]+=valor o valores
	else if (step == 1)
		sparse_mat_vec_mul2_sks(elmA, idxA, b, c,
			order);
	else;
}

//U*b
void sparse_mat_vec_mul1_sks(__global double* elmA,
	__global int* idxA, __global double* b,
	__global double* c, int order,
	__local double* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//índice local de hilo
	int tx = get_local_id(0);

	for (int row = bx; row < order; row += get_num_groups(0)) {

		//ubicación del elemento diagonal de la fila en elmA
		int dId = idxA[row];

		//número de elementos en la fila
		int nEl = (row == order - 1 ? 1 : idxA[row + 1] - dId);

		//valor donde se guardará una suma parcial
		double sum = 0.0;

		for (int x = tx; x < nEl; x += get_local_size(0)) {
			int cId = row + x;
			sum += elmA[dId + x] * b[cId];
		}
		partialSum[tx] = sum;

		//reducción en paralelo pare determinar la suma de
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
void sparse_mat_vec_mul2_sks(__global double* elmA,
	__global int* idxA, __global double* b,
	__global double* c, int order)
{
	//índice de grupo
	int bx = get_group_id(0);

	//índice local de hilo
	int tx = get_local_id(0);

	for (int col = bx; col < order; col += get_num_groups(0)) {
		//posición del elemento diagonal
		int dId = idxA[col];

		//número de elementos para una columna
		int nEl = (col == order - 1 ? 1 : idxA[col + 1] - dId);

		//elemento en b, por el que se multiplica
		double mul = b[col];

		for (int x = tx; x < nEl - 1; x += get_local_size(0)) {
			int rId = col + x + 1;
			double sum = mul * elmA[dId + x + 1];
			atom_add_double(&(c[rId]), sum);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

/*función atom_add para valores del tipo float en memoria
global
adaptación en OpenCL del código en:
https://docs.nvidia.com/cuda/archive/10.1/cuda-c-programming-guide/index.html
*/
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

/*función atom_add para valores del tipo double en memoria
global
adaptación en OpenCL del código en:
https://docs.nvidia.com/cuda/archive/10.1/cuda-c-programming-guide/index.html
*/
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
//código de sincronización
//------------------------
/*Código realizado con referencia al artículo:
Inter-Block GPU Communication via Fast Barrier Synchronization
Shucai Xiao and Wu-chun Feng
Department of Computer Science Virginia Tech
2009/9/19
Fig.9. Pág.6
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

/*Código realizado con referencia al artículo:
Inter-Block GPU Communication via Fast Barrier Synchronization
Shucai Xiao and Wu-chun Feng
Department of Computer Science Virginia Tech
2009/9/19
Fig.6. Pág.5
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
	// índice local de hilo
	int tlx = get_local_id(0);
	int tly = get_local_id(1);

	//índice de grupo
	int bx = get_group_id(0);
	int by = get_group_id(1);

	// índice global de hilo
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