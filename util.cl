//============================================
//revisión 0.0.0  01-12-2019, 18:10 VS 2017
//============================================
 
 /* se parte de dos vectores row y col
 que son los vectores de índices de fila y columna 
 respectivamente de una matriz almacenada en formato
 COO.
 el vector col debe tener sus elementos ordenados,
el vector row debe tener sus elementos ordenados para 
cada valor en col, por
ejemplo:

row col
1    1
2    1
3    1
1    2
2    2
3    2

se almacenará en index, la cantidad de elementos necesarios
a almacenar por columna en el formato SKS 
 */
 __kernel void sks_util_1(__global int* index,
	 __global int* row, __global int* col, int size)
{
    // indice global de hilo
    int tx = get_global_id(0);

	//cantidad total de hilos
	int ntx = get_local_size(0)*get_num_groups(0);

	//índice de fila y columna actual
	int idr;
	int idc;

	for (int x = tx; x < size; x += ntx) {
		//si dos elementos consecutivos de col son 
		//diferentes, indicará que se ha cambiado la 
		//columna, el elemento correspondiente en row será
		//almacenado en index
		idc = col[x];
		idr = row[x];

		if (x == size - 1) {
			index[idc] = 1;
		}
		else {
			if (idc != col[x + 1]) {
				//se almacena solo de la diagonal para abajo
				index[idc] = idr - idc + 1;
			}
		}
	}
}

/*las matrices equivalentes en formato SKS (sks y ptr)
y COO (nz, row, col), se copiaran los elementos de nz
en el lugar correspondiente de sks*/
 __kernel void sks_util_2(__global double* sks,
	 __global int* ptr, __global double* nz,
	 __global int* row, __global int* col, int size)
 {
	 // indice global de hilo
	 int tx = get_global_id(0);

	 //cantidad total de hilos
	 int ntx = get_local_size(0)*get_num_groups(0);

	 //índices actuales de fila y columna
	 int idr;
	 int idc;
	 int idp;//apuntador actual

	 for (int x = tx; x < size; x += ntx) {
		 //sobteniendo índice de fila y columna
		 idr = row[x];
		 idc = col[x];

		 //apuntador actual
		 idp = ptr[idc];

		 //almacenando en el lugar correspondiente
		 //solo si el elemento está bajo la diagonal
		 if (idr >= idc)
			 sks[idp + idr - idc] = nz[x];
	 }
 }

 __kernel void sks_util_2_sp(__global float* sks,
	 __global int* ptr, __global float* nz,
	 __global int* row, __global int* col, int size)
 {
	 // indice global de hilo
	 int tx = get_global_id(0);

	 //cantidad total de hilos
	 int ntx = get_local_size(0)*get_num_groups(0);

	 //índices actuales de fila y columna
	 int idr;
	 int idc;
	 int idp;//apuntador actual

	 for (int x = tx; x < size; x += ntx) {
		 //sobteniendo índice de fila y columna
		 idr = row[x];
		 idc = col[x];

		 //apuntador actual
		 idp = ptr[idc];

		 //almacenando en el lugar correspondiente
		 //solo si el elemento está bajo la diagonal
		 if (idr >= idc)
			 sks[idp + idr - idc] = nz[x];
	 }
 }

 /* se parte de dos vectores row y col
 que son los vectores de índices de fila y columna
 respectivamente de una matriz almacenada en formato
 COO.
 el vector col debe tener sus elementos ordenados,
el vector row debe tener sus elementos ordenados para
cada valor en col, por
ejemplo:

row col
1    1
2    1
3    1
1    2
2    2
3    2

se almacenará en index, el número de elementos por 
columna acumulada
 */
 __kernel void csc_util_1(__global int* index,
	 __global int* row, __global int* col, int size)
 {
	 // indice global de hilo
	 int tx = get_global_id(0);

	 //cantidad total de hilos
	 int ntx = get_local_size(0)*get_num_groups(0);

	 //índice de fila y columna actual
	 int idr;
	 int idc;

	 for (int x = tx; x < size; x += ntx) {
		 //si dos elementos consecutivos de col son 
		 //diferentes, indicará que se ha cambiado la 
		 //columna, el valor de x aumentado en uno
		 //será almacenado en index
		 idc = col[x];

		 if (x == size - 1) {
			 index[idc] = size;
		 }
		 else {
			 if (idc != col[x + 1]) {
				 //se almacena solo de la diagonal para abajo
				 index[idc] = x + 1;
			 }
		 }
	 }
 }

 /*buscará cuantos elementos deben eliminarse por columna
 para almacenar solo la parte triangular inferior,usando 
 el vector obtenido con la función csc_util_1()*/
 __kernel void csc_util_2(__global int* del,
	 __global int* index, __global int* row, int orden)
 {
	 // indice local de hilo
	 int tlx = get_local_id(0);
	 //indice de grupo
	 int bx = get_group_id(0);
	 //para almacenar un apuntador 
	 int ptr;
	 //para almacenar indice de fila actual
	 int idr;
	 //número de elementos en una columna
	 int nel;
	 //iteración para cubrir todas las columnas
	 for (int x = bx; x < orden; x += get_num_groups(0)) {
		 //consiguiendo apuntador hacia row par la columna 
		 ptr = (x == 0 ? 0 : index[x - 1]);
		 nel = index[x] - ptr;
		 //iteración para cubrir todos los elementos en 
		 //la columna (almacenados)
		 for (int y = tlx; y < nel; y += get_local_size(0)) {
			 idr = row[ptr + y];
			 //cuandp idr sea igual al índice de columna
			 //actual se habrá encontrado el elemento
			 //diagonal, y indicará cuantos elementos
			 //se debn eliminar en esta columna
			 if (idr == x)
				 del[x] = y;
		 }
	 }
 }

 /*apartir de los vectores coseguidos usando las dos
 funciones anteriores, se modificará el devuelto por
 csc_util_1, para obtener apuntadores al vector final
 csc, antes de poner como argumento el vector devuelto
 por csc_util_2, deberá acumularse en serie*/
 __kernel void csc_util_3(__global int* del,
	 __global int* ptr, int orden,
	 __local int* temp)
 {
	 // indice global de hilo
	 int tlx = get_local_id(0);

	 //cantidad total de hilos
	 int ntlx = get_local_size(0);

	 //índice a modificar
	 int idm;

	 //solo se utilizará solo un grupo, para evitar
	 //errores durante la escritura, la memoria local
	 //debe tener el mismo tamaño que un grupo
	 if (get_group_id(0) == 0) {
		 for (int x = tlx; x < orden; x += ntlx) {
			 //se modificará desde abajo
			 idm = orden - x - 1;
			 //almacenando en memoria local
			 temp[tlx] = (idm == 0 ? 0 :
				 ptr[idm - 1] - del[idm - 1]);
			 //sincronizando para asegurar que la
			 //escritura de memoria local ha sido
			 //completada
			 barrier(CLK_LOCAL_MEM_FENCE);

			 //escribiendo valores finales
			 if (idm == 0)
				 ptr[idm] = 0;
			 else
				 ptr[idm] = temp[tlx];
		 }
	 }
 }

 /*las matrices equivalentes en formato CSC (csc, rowc y ptr)
y COO (nz, row, col), se copiaran los elementos de nz y row
en el lugar correspondiente de csc y rowc.
ptr deberá haberse obtenido usando la función anterior, el vector 
obtenido por csc_util_2 debe acumularse y ponerse como
argumento
del formato COO solo se necesitan los argumentos nz y row*/
 __kernel void csc_util_4(__global double* csc,
	 __global int* rowc,__global int* ptr,
	 __global double* nz, __global int* row,
	 __global int* del, int orden)
 {
	 // indice local de hilo
	 int tlx = get_local_id(0);
	 //indice de grupo
	 int bx = get_group_id(0);
	 //para almacenar apuntadores en csc, rowc y nz
	 //respectivamente correspondiente a una columna
	 int ptr_csc;
	 int ptr_rowc;
	 int ptr_nz;
	 //número de elementos a almacenar por columna
	 int nel;
	 //valo actual en del
	 int vdel;
	 //númer de elementos a eliminar
	 int ddel;
	 //iteración para cubrir todas las columnas
	 for (int x = bx; x < orden; x += get_num_groups(0)) {
		 //obteniendo apuntadores
		 ptr_csc = ptr[x];
		 ptr_nz = (x == 0 ? 0 : ptr_csc + del[x - 1]);
		 ptr_rowc = ptr_csc - x;
		 //valor actual en del
		 vdel = del[x];
		 //número de elementos a no almacenar de nz por
		 //columna
		 ddel = (x == 0 ? 0 : vdel - del[x - 1]);
		 //número de elementos a almacenar por columna
		 nel = (x == orden - 1 ? 1 :
			 ptr[x + 1] - ptr_csc);
		 //iteración para cubrir todos los elementos en 
		 //la columna (almacenados)
		 for (int y = tlx; y < nel; y += get_local_size(0)) {
	
			 //almacenando en csc
			 csc[ptr_csc + y] = nz[ptr_nz + ddel + y];
			 //almacenando en rowc
			 //almacenará un índice adelantado, omitiendo
			 //el último
			 if (y != nel - 1)
				 rowc[ptr_rowc + y] =
				 row[ptr_nz + ddel + y + 1];
		 }
	 }
 }

 __kernel void csc_util_4_sp(__global float* csc,
	 __global int* rowc, __global int* ptr,
	 __global float* nz, __global int* row,
	 __global int* del, int orden)
 {
	 // indice local de hilo
	 int tlx = get_local_id(0);
	 //indice de grupo
	 int bx = get_group_id(0);
	 //para almacenar apuntadores en csc, rowc y nz
	 //respectivamente correspondiente a una columna
	 int ptr_csc;
	 int ptr_rowc;
	 int ptr_nz;
	 //número de elementos a almacenar por columna
	 int nel;
	 //valo actual en del
	 int vdel;
	 //númer de elementos a eliminar
	 int ddel;
	 //iteración para cubrir todas las columnas
	 for (int x = bx; x < orden; x += get_num_groups(0)) {
		 //obteniendo apuntadores
		 ptr_csc = ptr[x];
		 ptr_nz = (x == 0 ? 0 : ptr_csc + del[x - 1]);
		 ptr_rowc = ptr_csc - x;
		 //valor actual en del
		 vdel = del[x];
		 //número de elementos a no almacenar de nz por
		 //columna
		 ddel = (x == 0 ? 0 : vdel - del[x - 1]);
		 //número de elementos a almacenar por columna
		 nel = (x == orden - 1 ? 1 :
			 ptr[x + 1] - ptr_csc);
		 //iteración para cubrir todos los elementos en 
		 //la columna (almacenados)
		 for (int y = tlx; y < nel; y += get_local_size(0)) {

			 //almacenando en csc
			 csc[ptr_csc + y] = nz[ptr_nz + ddel + y];
			 //almacenando en rowc
			 //almacenará un índice adelantado, omitiendo
			 //el último
			 if (y != nel - 1)
				 rowc[ptr_rowc + y] =
				 row[ptr_nz + ddel + y + 1];
		 }
	 }
 }