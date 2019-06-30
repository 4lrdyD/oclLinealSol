/*
=========================================
revisión 0.0.1 30-06-2019, 00:40 VS 2017
=========================================
*/
/* Solución de un sistema de ecuaciones, usando 
la factorización LDLT: para un sistema de la forma:
A*x=B donde la incógnita es x; A es una matriz cuadrada
simétrica de orden n y B es un vector
de longitud n, una vez hecha la factorización será
necesario resolver el sistema equivalente teniendo 
la matriz triangular generada por la factorización

Ver:
Técnicas de Cálculo para Sistemas de Ecuaciones,
Programación Lineal y Programación Entera
José Luis de la Fuente O’Connor
Cap.1, 1.5. Factorización de matrices simétricas
1.5.1 Factorización LDLT
Pág. 40
 */
void __gpu_sync(int goalVal,
	__global volatile int * Arrayin, __global volatile int * Arrayout);

void __gpu_sync1(int goalVal,
	__global volatile int * g_mutex);

//float
void ldlt_1_csp(__global float* A, int order, int step);
void ldlt_2_csp(__global float* A, int order, int step);
void ldlt_3_csp(__global float* L, __global float* b, int order,
	int step);
void ldlt_4_csp(__global float* L, __global float* b, int order);
void ldlt_5_csp(__global float* L, __global float* y, int order,
	int step, __local float* partialSum);
void sparse_fill_csp(__global float* elmA, __global int* colA,
	__global int* rowA, __global float* elmL, __global int* colL,
	__global int* rowL, int size_elmA);

//
void ldlt_sparse_1_csp(__global float* elmL, __global int* colL,
	int order, int step);

void ldlt_sparse_2_csp(__global float* elmL, __global int* colL,
	__global int* rowL, int order, int step);

void ldlt_sparse_3_csp(__global float* elmL,
	__global int* colL, __global int* rowL, __global float* b,
	int order, int step);

void ldlt_sparse_4_csp(__global float* elmL,
	__global int* colL, __global float* b, int order);

void ldlt_sparse_5_csp(__global float* elmL,
	__global int* colL, __global int* rowL, __global float* y,
	int order, int step, __local float* partialSum);

//
void ldlt_sparse_1_sks_sp(__global float* elmL,
	__global int* idxL, int order, int step);

void ldlt_sparse_2_sks_sp(__global float* elmL,
	__global int* idxL, int order, int step);

void ldlt_sparse_3_sks_sp(__global float* elmL,
	__global int* idxL, __global float* b, int order,
	int step);
void ldlt_sparse_4_sks_sp(__global float* elmL,
	__global int* idxL, __global float* b, int order);

void ldlt_sparse_5_sks_sp(__global float* elmL,
	__global int* idxL, __global float* y,
	int order, int step, __local float* partialSum);

//double
void ldlt_1_c(__global double* A, int order, int step);
void ldlt_2_c(__global double* A, int order, int step);
void ldlt_3_c(__global double* L, __global double* b, int order,
	int step);
void ldlt_4_c(__global double* L, __global double* b, int order);
void ldlt_5_c(__global double* L, __global double* y, int order,
	int step, __local double* partialSum);

//------------------------
//simple precisión (float)
//------------------------

/*Kernel que concibe los elementos de A, como ordenados por 
columna, simple precisión, la factorización se hará inplace,
al final de la factorización la matriz A contendra los elementos
de L en la parte triangular inferior excepto en su diagonal
(estos son implicitamente 1 en L y no necesitan almacenarse), y
los elementos de D en su diagonal:

d1  
l21 d2
l31 l32 d3
l41 l42 l43 d4
.    .	 .   .	.	
.    .   .   .	.	.
.    .   .   .	.	.	.
*/
__kernel void
ldlt_c_sp(__global float* A, __global float* b,
	__local float* partialSum, int order, int step, int sstep)
{
	//la iteración principal se realiza en el host

	//se modificará una sola columna
	if (sstep == 0) {
		ldlt_1_csp(A, order, step);
	}
	//se modificará las columnas siguientes a la columna
	//implicada
	else if (sstep==1) {
		ldlt_2_csp(A, order, step);
	}

	///hasta aqui la factorización esta concluida, la parte
	//triangular inferior de A contendrá la factorización LDLT
	//es decir L y D
	//desde aqui se procederá a solucionar el sistema 
	//L*D*transpose(L)x=b

	//se resolverá el sistema Ly=b, y=D*transpose(L)*x
	else if (sstep==2){
		ldlt_3_csp(A, b, order, step);
	}

	//se resolverá el sistema Dz=y, z=transpose(L)*x
	else if (sstep == 3) {
		ldlt_4_csp(A, b, order);
	}

	//se resolverá el sistema transpose(L)*x=z
	//"z" ya debe haberse obtenido usando la función anterior
	else{
		ldlt_5_csp(A, b, order, step, partialSum);
	}
}

/*
paso j(1) de la factorización LDLT inplace (es decir
la matriz A será modificada) en este paso, se modifica
la columna j.

Se modifican los elementos de la columna j a partir de la fila
j + 1.

A(n,j)=A(n,j)/A(j,j) n>j

se asume un almacenamiento por columna

se aplica un método de programación similar al código usado 
en la factorización de cholesky.
*/
void ldlt_1_csp(__global float* A, int order, int step)
{

	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//columna base
	__global float* colB = A + step * order;

	//elemento A(step,step)
	float _r = colB[step];

	//cada hilo se encargará de modificar un elemento 
	//de la columna
	for (int x = tx + step + 1; x < order; x += ntx) {
		colB[x] = native_divide(colB[x], _r);
	}
}

/*
paso j(2) de la factorización LDLT inplace (es decir
la matriz A será modificada) para este paso, se modifica
las columnas siguientes a la columna j.

para una columna s, s>j, se modifican las filas n>=s

A(n,s)=A(n,s)-A(s,j)*A(n,j)*A(j,j) s>j, n>=s

se asume un almacenamiento por columna

se aplica un método de programación similar al código usado
en la factorización de cholesky.
*/
void ldlt_2_csp(__global float* A, int order, int step)
{
	//índice de grupo
	int bx = get_group_id(0);

	//índice local de hilo
	int tx = get_local_id(0);

	//columna base
	__global float* colB = A + step * order;

	//elemento A(step,step)
	float _r = colB[step];

	//parámetros de apoyo
	float _k;

	//en este caso, cada grupo se encargará de las 
	//operaciones sobre todos los elementos de una 
	//columna
	for (int y = bx + step + 1; y < order; y += get_num_groups(0)) {
		_k = _r * colB[y];
		__global float* colI = A + y * order;

		for (int x = tx + y; x < order;
			x += get_local_size(0)) {

			colI[x] -= _k * colB[x];
		}
	}
}

/*
Inicia la solución del sistema Ax=b teniendo la matriz trangular
inferior L y la matriz diagonal D (obtenido tras la 
factorización de LDLT), tal que A= L*D*transpose(L)

se usará esta función para resolver el sistema Ly=b con
y=D*transpose(L)*x que será resuelto posteriormente con la función
ldlt_4_csp

la factorización LDLT modificó la matriz A tal que en su parte
triangular inferior se encuentran los elementos tanto de L como
de D, para esta función solo se usa L, por tanto solo se usará
la parte triangular inferior sin la diagonal, debe considerarse
de que los elementos de L son 1 implicitamente.

para un paso j, modificará el vector b tal que:

b[n]=b[n]-b[j]*L[n,j] n>j

este proceso es equivalente a usar el método de Gauss_Jordan
en matrices triangulares inferiores, en este caso es posible 
dejar intacta la matriz L modificando solo el vector b, a pesar
de que para un paso j la idea central es convertir en cero toda
la columna j de A, estas operaciones pueden obviarse ya que no se 
necesita el valor final, por lo tanto lo que importa solo es el
trabajo que se realiza sobre el vector b

se asume un almacenamiento por columna
*/
void ldlt_3_csp(__global float* L, __global float* b, int order,
	int step)
{

	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//columna de L implicada
	__global float* colB = L + order * step;

	for (int x = tx + step + 1; x < order; x += ntx) {
		b[x] -= b[step] * colB[x];
	}
}

/*
divide el vector b, entre la diagonal de D
b[j]=b[j]/D[j,j], equivalente a resolver el sistema
Dz=y, con z=transpose(L)*x que se resolverá con ldlt_5_csp

*/
void ldlt_4_csp(__global float* D, __global float* b, int order)
{

	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//parametros de apoyo
	int diag;

	for (int x = tx; x < order; x += ntx) {
		//ubicación del elemento diagonal en memoria
		diag = x * order + x;
		b[x] = native_divide(b[x], D[diag]);
	}
}

/*
Inicia la solución del sistema Ax=b teniendo la matriz trangular
inferior L (obtenido tras la factorización LDLT),
tal que A= L*D*transpose(L)

Con esta función se iniciará la solución del sistema 
transpose(L)*x=z, "z" debe haberse obtenido anteriormente usando
la función ldlt_4_csp, se toma en cuenta que los elementos 
diagonales de L son 1 implícitamente (no está almacenado)

para un paso j modificará el elemento b[i] tal que i=n-j:

b[i]=b[i]-sumatoria{j=i+1,n}(z[j]*L[j,i])

este valor será la solución final x[i] del sistema Ax=b

se asume un almacenamiento por fila, esto para poder usar la
misma matriz L sin necesidad de transponerla, además se usará
solo un bloque.

*/
void ldlt_5_csp(__global float* L, __global float* z, int order,
	int step, __local float* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//índice global de hilo
	int tx = get_local_id(0);

	//número total de hilos
	int ntx = get_local_size(0);

	//indice de la fila de L implicada
	int fL = order - step - 1;

	//fila de L implicada, data
	__global float* filB = L + fL * order;

	//valor donde se guardará una suma parcial
	float sum = 0.0f;

	//solo se usará un bloque
	if (bx == 0) {
		for (int x = tx + fL + 1; x < order; x += ntx) {
			sum += filB[x] * z[x];
		}
		partialSum[tx] = sum;

		//reducción en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = ntx / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			z[fL] -= partialSum[0];
	}
}


/*Aplicado a matrices dispersas y simétricas almacenadas
en formato SCS (Store Compresed Colum):

Sea una matriz representada en su forma densa por A.
esta matriz en su forma dispersa en formato CSC será 
representada por 3 vectores.

*elmA: Vector de elementos distintos de cero, guardados
columna a columna.

*colA: Vector de índices, donde cada elemento indica la
ubicación en elmA del primer elemento distinto de cero en
cada columna de A

*rowA: Vector de índices en donde se guardan los índices
de fila en A de los elementos en elmA exceptuando los elementos
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
	LDLt (ya que los elementos del L original son unos, no 
	será necesario almacenarlos, se	almacenará en vez de 
	estos los elementos de D), 
	elmL debe ser del tamaño adecuado para guardar
	todos los elementos distintos de cero de L, colL y rowL
	deben estar ya con sus valores finales, estos deben
	obtenerse mediante una factorización simbólica, tomar en
	cuenta que la factorización LDLt simbólica es la misma
	que se obtiene de una factorización de cholesky
	simbólica*/

//-------------------------------
__kernel void
ldlt_sparse_c_sp(__global float* elmA, __global int* colA,
	__global int* rowA, __global float* elmL, __global int* colL,
	__global int* rowL, int size_elmA, int size_colA,
	__global float* b, __local float* partialSum, int step,
	int sstep)
{
	if (sstep == -1)
		sparse_fill_csp(elmA, colA, rowA, elmL, colL,
			rowL, size_elmA);
	//la iteración principal se realiza en el host

	//se modificará una sola columna
	else if (sstep == 0)
		ldlt_sparse_1_csp(elmL, colL, size_colA, step);

	//se modificará las columnas siguientes a la columna
	//implicada
	else if (sstep == 1)
		ldlt_sparse_2_csp(elmL, colL, rowL, size_colA, step);

	//hasta aqui la factorización esta concluida, la parte
	//triangular inferior de A contendrá la factorización
	//LDLt es decir L y D
	//desde aqui se procederá a solucionar el sistema 
	//L*D*transpose(L)x=b

	//se resolverá el sistema Ly=b, y=D*transpose(L)*x
	else if (sstep == 2)
		ldlt_sparse_3_csp(elmL, colL, rowL, b, size_colA, step);
	
	//se resolverá el sistema Dz=y, z=transpose(L)*x
	else if (sstep == 3)
		ldlt_sparse_4_csp(elmL, colL, b, size_colA);

	//se resolverá el sistema transpose(L)*x=z
	//"z" ya debe haberse obtenido usando la función anterior
	else
		ldlt_sparse_5_csp(elmL, colL, rowL, b, size_colA, step,
			partialSum);
}

/*dada dos matrices dispersas A y L, copia los elementos de
elmA en el lugar correspondiente de elmL.
Se asume que los índices de rowA están incluidos en
rowL. Es decir, en su forma densa, todos
los lugares diferentes de cero en A, también son diferentes de
cero en L, pero no viceversa*/

void sparse_fill_csp(__global float* elmA, __global int* colA,
	__global int* rowA, __global float* elmL, __global int* colL,
	__global int* rowL, int size_elmA)
{
	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < size_elmA; x += ntx) {
		//variable donde se guardará el índice de columna y de fila
		//respectivamente del elemento elmA[tx]
		int col = 0;
		int row = 0;
		//variable que controlará la salida del bucle.
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
		//se cuenta con el índice de fila y de columna (row y col)
		//ahora necesitamos ubicar este elemento en elmL

		int posf;//posición final en elmL

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

void ldlt_sparse_1_csp(__global float* elmL, __global int* colL,
	int order, int step)
{
	//es procedimiento solo deberá realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la última 
	//columna tendrá índice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//obteniendo la ubicación del primer elemento 
		//de la columna determinada por step
		int pos = colL[step];

		//posición siguiente
		int pos_nxt = colL[step + 1];

		//índice global de hilo
		int tx = get_global_id(0);

		//número total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//elemento diagonal. El primer
		//elemento guardado en elmA para cada columna es el
		//elemento diagonal
		float _r = elmL[pos];

		//cada hilo se encargará de modificar un elemento 
		//de la columna
		for (int x = tx + pos + 1; x < pos_nxt; x += ntx) {
			elmL[x] = native_divide(elmL[x], _r);
		}
	}
}

void ldlt_sparse_2_csp(__global float* elmL, __global int* colL,
	__global int* rowL, int order, int step)
{
	//es procedimiento solo deberá realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la última 
	//columna tendrá índice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//índice local de grupo
		int bx = get_group_id(0);

		//índice local de hilo
		int tx = get_local_id(0);

		//posición del primer elemento de la columna de elmL
		int pos_elmL = colL[step];

		//elemento diagonal
		float diag = elmL[pos_elmL];

		//posición del primer elemento de la columna en rowL
		int pos_rowL = pos_elmL - step;

		//número de elementos en rowL para la columna
		int nEl = colL[step + 1] - pos_elmL - 1;

		if (nEl > 0) {

			for (int b = bx; b < nEl; b += get_num_groups(0)) {
				//indice de la columna a modificar
				int colId = rowL[pos_rowL + b];

				//primer factor
				float k1 = elmL[pos_elmL + b + 1];

				for (int x = b + tx; x < nEl; x += get_local_size(0)) {

					//índice de la fila en la columna a modificar
					int rowId = rowL[pos_rowL + x];

					//segundo factor
					float k2 = elmL[pos_elmL + x + 1];

					int posf;//posición final en elmL

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

					elmL[posf] -= diag * k1 * k2;
				}
			}
		}
	}
}

void ldlt_sparse_3_csp(__global float* elmL,
	__global int* colL, __global int* rowL, __global float* b,
	int order, int step)
{
	if (step < order - 1) {
		//índice global de hilo
		int tx = get_global_id(0);

		//número total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//posición del elemento diagonal
		int diag = colL[step];

		//número de elementos para una columna
		int nEl = colL[step + 1] - diag;

		//deben haber elementos aparte del elemento diagonal
		if (nEl > 1) {

			//elemento diagonal y elemento correspondiente en b
			float diag_elm = elmL[diag];
			float elmb = b[step];

			//posición en rowL del primer elemento fuera de la
			//diagonal (índice)
			int pos_rowL = diag - step;

			for (int x = tx; x < nEl - 1; x += ntx) {
				int rowId = rowL[pos_rowL + x];
				b[rowId] -= elmb * elmL[diag + x + 1];
			}
		}
	}
}

void ldlt_sparse_4_csp(__global float* elmL,
	__global int* colL, __global float* b, int order)
{

	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		//ubicación del elemento diagonal en elmL
		int diag = colL[x];
		b[x] = native_divide(b[x], elmL[diag]);
	}
}

void ldlt_sparse_5_csp(__global float* elmL,
	__global int* colL, __global int* rowL, __global float* y,
	int order, int step, __local float* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//índice global de hilo
	int tx = get_local_id(0);

	//número total de hilos
	int ntx = get_local_size(0);

	//indice de la fila de L implicada
	int fL = order - step - 1;

	//ubicación de elemento diagonal de la fila en elmL
	int diagId = colL[fL];

	//número de elementos en la fila
	int nEl = (fL == order - 1 ? 1 : colL[fL + 1] - diagId);

	//ubicación del índice del primer elemento fuera de la
	//diagonal en rowL
	int pos_rowL = diagId - fL;

	//valor donde se guardará una suma parcial
	float sum = 0.0f;

	//solo se usará un bloque
	if (bx == 0) {
		for (int x = tx; x < nEl - 1; x += ntx) {
			int rowId = rowL[pos_rowL + x];
			sum += elmL[diagId + x + 1] * y[rowId];
		}
		partialSum[tx] = sum;

		//reducción en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = ntx / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			y[fL] -= partialSum[0];
	}
}


/*Aplicado a matrices dispersas y simétricas almacenadas
en formato SKS (Skyline Storage):

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

//-----------------------------------
//elmL es elmA o una copia del mismo
__kernel void
ldlt_sparse_sks_sp(__global float* elmL,
	__global int* idxL, int size_elmL, int size_idxL,
	__global float* b, __local float* partialSum, int step,
	int sstep)
{
	//la iteración principal se realiza en el host

	//se modificará una sola columna
	if (sstep == 0)
		ldlt_sparse_1_sks_sp(elmL, idxL, size_idxL, step);

	//se modificará las columnas siguientes a la columna
	//implicada
	else if (sstep == 1)
		ldlt_sparse_2_sks_sp(elmL, idxL, size_idxL, step);

	//hasta aqui la factorización esta concluida, la parte
	//triangular inferior de A contendrá la factorización
	//LDLt es decir L y D
	//desde aqui se procederá a solucionar el sistema 
	//L*D*transpose(L)x=b

	//se resolverá el sistema Ly=b, y=D*transpose(L)*x
	else if (sstep == 2)
		ldlt_sparse_3_sks_sp(elmL, idxL, b, size_idxL,
		step);

	//se resolverá el sistema Dz=y, z=transpose(L)*x
	else if (sstep == 3)
		ldlt_sparse_4_sks_sp(elmL, idxL, b, size_idxL);

	//se resolverá el sistema transpose(L)*x=z
	//"z" ya debe haberse obtenido usando la función anterior
	else
		ldlt_sparse_5_sks_sp(elmL, idxL, b, size_idxL,
		step, partialSum);
}

void ldlt_sparse_1_sks_sp(__global float* elmL, 
	__global int* idxL, int order, int step)
{
	//es procedimiento solo deberá realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la última 
	//columna tendrá índice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//obteniendo la ubicación del primer elemento 
		//de la columna determinada por step
		int pos = idxL[step];

		//posición siguiente
		int pos_nxt = idxL[step + 1];

		//índice global de hilo
		int tx = get_global_id(0);

		//número total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//elemento diagonal. El primer
		//elemento guardado en elmA para cada columna es el
		//elemento diagonal
		float _r = elmL[pos];

		//cada hilo se encargará de modificar un elemento 
		//de la columna
		for (int x = tx + pos + 1; x < pos_nxt; x += ntx) {
			elmL[x] = native_divide(elmL[x], _r);
		}
	}
}

void ldlt_sparse_2_sks_sp(__global float* elmL,
	__global int* idxL, int order, int step)
{
	//este procedimiento solo deberá realizarse hasta 
	//la columna order-2 
	//Siendo order el orden de la matriz, la última 
	//columna tendrá índice order-1, donde solo contiene
	//el elemento diagonal
	if (step < order - 1) {
		//índice local de grupo
		int bx = get_group_id(0);

		//índice local de hilo
		int tx = get_local_id(0);

		//posición del primer elemento de la columna de elmL
		int pos_elmL = idxL[step];

		//elemento diagonal
		float diag = elmL[pos_elmL];

		//número de elementos en elmL para la columna
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

					//posición final en elmL
					int	posf = idxL[colId] + x - b;

					elmL[posf] -= diag * k1 * k2;
				}
			}
		}
	}
}

void ldlt_sparse_3_sks_sp(__global float* elmL,
	__global int* idxL, __global float* b, int order,
	int step)
{
	if (step < order - 1) {
		//índice global de hilo
		int tx = get_global_id(0);

		//número total de hilos
		int ntx = get_num_groups(0)*get_local_size(0);

		//posición del elemento diagonal
		int diag = idxL[step];

		//número de elementos para una columna
		int nEl = idxL[step + 1] - diag;

		//deben haber elementos aparte del elemento diagonal
		if (nEl > 1) {

			//elemento diagonal y elemento correspondiente en b
			float diag_elm = elmL[diag];
			float elmb = b[step];

			for (int x = tx; x < nEl - 1; x += ntx) {
				int rowId = step + x + 1;
				b[rowId] -= elmb * elmL[diag + x + 1];
			}
		}
	}
}

void ldlt_sparse_4_sks_sp(__global float* elmL,
	__global int* idxL, __global float* b, int order)
{

	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	for (int x = tx; x < order; x += ntx) {
		//ubicación del elemento diagonal en elmL
		int diag = idxL[x];
		b[x] = native_divide(b[x], elmL[diag]);
	}
}

void ldlt_sparse_5_sks_sp(__global float* elmL,
	__global int* idxL,__global float* y,
	int order, int step, __local float* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//índice global de hilo
	int tx = get_local_id(0);

	//número total de hilos
	int ntx = get_local_size(0);

	//indice de la fila de L implicada
	int fL = order - step - 1;

	//ubicación del elemento diagonal de la fila en elmL
	int diagId = idxL[fL];

	//número de elementos en la fila
	int nEl = (fL == order - 1 ? 1 : idxL[fL + 1] - diagId);

	//valor donde se guardará una suma parcial
	float sum = 0.0f;

	//solo se usará un bloque
	if (bx == 0) {
		for (int x = tx; x < nEl - 1; x += ntx) {
			int rowId = fL + x + 1;
			sum += elmL[diagId + x + 1] * y[rowId];
		}
		partialSum[tx] = sum;

		//reducción en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = ntx / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			y[fL] -= partialSum[0];
	}
}

//------------------------
//precisión doble (double)
//------------------------

/*Kernel que concibe los elementos de A, como ordenados por
columna simple precisión, la factorización se hará inplace,
al final de la factorización la matriz A contendra los elementos
de L en la parte triangular inferior excepto en su diagonal
(estos son implicitamente 1 en L y no necesitan almacenarse), y
los elementos de D en su diagonal:

d1
l21 d2
l31 l32 d3
l41 l42 l43 d4
.    .	 .   .	.
.    .   .   .	.	.
.    .   .   .	.	.	.
*/
__kernel void
ldlt_c(__global double* A, __global double* b,
	__local double* partialSum, int order, int step, int sstep)
{
	//la iteración principal se realiza en el host

	//se modificará una sola columna
	if (sstep == 0) {
		ldlt_1_c(A, order, step);
	}
	//se modificará las columnas siguientes a la columna
	//implicada
	else if (sstep == 1) {
		ldlt_2_c(A, order, step);
	}

	///hasta aqui la factorización esta concluida, la parte
	//triangular inferior de A contendrá la factorización LDLT
	//es decir L y D
	//desde aqui se procederá a solucionar el sistema 
	//L*D*transpose(L)x=b

	//se resolverá el sistema Ly=b, y=D*transpose(L)*x
	else if (sstep == 2) {
		ldlt_3_c(A, b, order, step);
	}

	//se resolverá el sistema Dz=y, z=transpose(L)*x
	else if (sstep == 3) {
		ldlt_4_c(A, b, order);
	}

	//se resolverá el sistema transpose(L)*x=z
	//"z" ya debe haberse obtenido usando la función anterior
	else {
		ldlt_5_c(A, b, order, step, partialSum);
	}
}

/*
paso j(1) de la factorización LDLT inplace (es decir
la matriz A será modificada) en este paso, se modifica
la columna j.

Se modifican los elementos de la columna j a partir de la fila
j + 1.

A(n,j)=A(n,j)/A(j,j) n>j

se asume un almacenamiento por columna

se aplica un método de programación similar al código usado
en la factorización de cholesky.
*/
void ldlt_1_c(__global double* A, int order, int step)
{

	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//columna base
	__global double* colB = A + step * order;

	//elemento A(step,step)
	double _r = colB[step];

	//cada hilo se encargará de modificar un elemento 
	//de la columna
	for (int x = tx + step + 1; x < order; x += ntx) {
		colB[x] = colB[x] / _r;
	}
}

/*
paso j(2) de la factorización LDLT inplace (es decir
la matriz A será modificada) para este paso, se modifica
las columnas siguientes a la columna j.

para una columna s, s>j, se modifican las filas n>=s

A(n,s)=A(n,s)-A(s,j)*A(n,j)*A(j,j) s>j, n>=s

se asume un almacenamiento por columna

se aplica un método de programación similar al código usado
en la factorización de cholesky.
*/
void ldlt_2_c(__global double* A, int order, int step)
{
	//índice de grupo
	int bx = get_group_id(0);

	//índice local de hilo
	int tx = get_local_id(0);

	//columna base
	__global double* colB = A + step * order;

	//elemento A(step,step)
	double _r = colB[step];

	//parámetros de apoyo
	double _k;

	//en este caso, cada grupo se encargará de las 
	//operaciones sobre todos los elementos de una 
	//columna
	for (int y = bx + step + 1; y < order; y += get_num_groups(0)) {
		_k = _r * colB[y];
		__global double* colI = A + y * order;

		for (int x = tx + y; x < order;
			x += get_local_size(0)) {

			colI[x] -= _k * colB[x];
		}
	}
}

/*
Inicia la solución del sistema Ax=b teniendo la matriz trangular
inferior L y la matriz diagonal D (obtenido tras la
factorización de LDLT), tal que A= L*D*transpose(L)

se usará esta función para resolver el sistema Ly=b con
y=D*transpose(L)*x que será resuelto posteriormente con la función
ldlt_4_csp

la factorización LDLT modificó la matriz A tal que en su parte
triangular inferior se encuentran los elementos tanto de L como
de D, para esta función solo se usa L, por tanto solo se usará
la parte triangular inferior sin la diagonal, debe considerarse
de que los elementos de L son 1 implicitamente.

para un paso j, modificará el vector b tal que:

b[n]=b[n]-b[j]*L[n,j] n>j

este proceso es equivalente a usar el método de Gauss_Jordan
en matrices triangulares inferiores, en este caso es posible
dejar intacta la matriz L modificando solo el vector b, a pesar
de que para un paso j la idea central es convertir en cero toda
la columna j de A, estas operaciones pueden obviarse ya que no se
necesita el valor final, por lo tanto lo que importa solo es el
trabajo que se realiza sobre el vector b

se asume un almacenamiento por columna
*/
void ldlt_3_c(__global double* L, __global double* b, int order,
	int step)
{

	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//columna de L implicada
	__global double* colB = L + order * step;

	for (int x = tx + step + 1; x < order; x += ntx) {
		b[x] -= b[step] * colB[x];
	}
}

/*
divide el vector b, entre la diagonal de D
b[j]=b[j]/D[j,j], equivalente a resolver el sistema
Dz=y, con z=transpose(L)*x que se resolverá con ldlt_5_csp

*/
void ldlt_4_c(__global double* D, __global double* b, int order)
{

	//índice global de hilo
	int tx = get_global_id(0);

	//número total de hilos
	int ntx = get_num_groups(0)*get_local_size(0);

	//parametros de apoyo
	int diag;

	for (int x = tx; x < order; x += ntx) {
		//ubicación del elemento diagonal en memoria
		diag = x * order + x;
		b[x] = b[x] / D[diag];
	}
}

/*
Inicia la solución del sistema Ax=b teniendo la matriz trangular
inferior L (obtenido tras la factorización LDLT),
tal que A= L*D*transpose(L)

Con esta función se iniciará la solución del sistema
transpose(L)*x=z, "z" debe haberse obtenido anteriormente usando
la función ldlt_4_csp, se toma en cuenta que los elementos
diagonales de L son 1 implícitamente (no está almacenado)

para un paso j modificará el elemento b[i] tal que i=n-j:

b[i]=b[i]-sumatoria{j=i+1,n}(z[j]*L[j,i])

este valor será la solución final x[i] del sistema Ax=b

se asume un almacenamiento por fila, esto para poder usar la
misma matriz L sin necesidad de transponerla, además se usará
solo un bloque.

*/
void ldlt_5_c(__global double* L, __global double* z, int order,
	int step, __local double* partialSum)
{
	//indice de bloque
	int bx = get_group_id(0);

	//índice global de hilo
	int tx = get_local_id(0);

	//número total de hilos
	int ntx = get_local_size(0);

	//indice de la fila de L implicada
	int fL = order - step - 1;

	//fila de L implicada, data
	__global double* filB = L + fL * order;

	//valor donde se guardará una suma parcial
	double sum = 0.0f;

	//solo se usará un bloque
	if (bx == 0) {
		for (int x = tx + fL + 1; x < order; x += ntx) {
			sum += filB[x] * z[x];
		}
		partialSum[tx] = sum;

		//reducción en paralelo pare determinar la suma de
		//todos los elementos de partialSum
		for (int stride = ntx / 2; stride > 0; stride /= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (tx < stride) {
				partialSum[tx] += partialSum[tx + stride];
			}
		}
		if (tx == 0)
			z[fL] -= partialSum[0];
	}
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
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
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
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

/*Kernel de prueba*/
__kernel void
prueba_(__global double* C, __global double* A, __global double* B,
	__local double As[BLOCK_SIZE][BLOCK_SIZE], int gsize)
{
	// índice local de hilo
	int tlx = get_local_id(0);
	int tly = get_local_id(1);

	// índice global de hilo
	int tgx = get_global_id(0);
	int tgy = get_global_id(1);

	if (tgx >= gsize || tgy >= gsize)
	{
		return;
	}
	//índice del primer elemento en una fila
	int Rfirst = tgy * gsize;

	//elemento diagonal
	double diag;

	//factor de multiplicación
	double fm;


	A[Rfirst + tgx] = tlx*10+tly;

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