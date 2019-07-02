//==========================================
//revisión 0.9.3 02-07-2019, 00:00 VS 2017
//==========================================

#include "Header.h"
#include "Common.h"
char * matrixMul_source=AFire::kernel_src("Common.h",
	"matrixMul.cl");
char * GJordan_source = AFire::kernel_src("Common.h",
	"Gauss_Jordan.cl");
char * Cholesky_source = AFire::kernel_src("Common.h",
	"Cholesky.cl");
char * ldlt_source = AFire::kernel_src("Common.h", "ldlt.cl");
char * gc_source = AFire::kernel_src("Common.h",
	"Gradiente_conjugado.cl");

using namespace af;

//----------
//utilidades
//----------
void AFire::copy(const af::array &A, af::array &B,size_t length) {

	// 2. Obtain the device, context, and queue used by ArrayFire	
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();
	// 3. Obtain cl_mem references to af::array objects
	cl_mem * d_A = A.device<cl_mem>();
	cl_mem * d_B = B.device<cl_mem>();
	
	// 4. Load, build, and use your kernels.
	//    For the sake of readability, we have omitted error checking.
	int status = CL_SUCCESS;
	// A simple copy kernel, uses C++11 syntax for multi-line strings.
	const char * kernel_name = "copy_kernel";
	const char * source = R"(
        void __kernel
        copy_kernel(__global float * gA, __global float * gB)
        {
            int id = get_global_id(0);
            gB[id] = gA[id];
        }
    )";
	// Create the program, build the executable, and extract the entry point
	// for the kernel.
	cl_program program = clCreateProgramWithSource(af_context, 1, &source, NULL, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);
	cl_kernel kernel = clCreateKernel(program, kernel_name, &status);
	// Set arguments and launch your kernels
	clSetKernelArg(kernel, 0, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), d_B);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, NULL, &length, NULL, 0, NULL, NULL);
	// 5. Return control of af::array memory to ArrayFire
	A.unlock();
	B.unlock();
	// ... resume ArrayFire operations
	// Because the device pointers, d_x and d_y, were returned to ArrayFire's
	// control by the unlock function, there is no need to free them using
	// clReleaseMemObject()
}

void AFire::sumaaf(af_array* out , af_array dA, af_array dB) {
	
	//to store the result
	af_array dC;
	af_copy_array(&dC, dA);

	// 2. Obtain the device, context, and queue used by ArrayFire	
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2], &_order[3], dA);
	size_t order = _order[0];

	int status = CL_SUCCESS;

	// 3. Obtain cl_mem references to af_array objects
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(float) * order,
		NULL, &status);
	af_get_device_ptr((void**)d_A, dA);

	cl_mem *d_B = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(float) * order,
		NULL, &status);
	af_get_device_ptr((void**)d_B, dB);

	cl_mem *d_C = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_WRITE_ONLY, sizeof(float) * order,
		NULL, &status);
	af_get_device_ptr((void**)d_C, dC);

	// 4. Load, build, and use your kernels.
	//    For the sake of readability, we have omitted error checking.
	// A simple sum kernel, uses C++11 syntax for multi-line strings.
	const char * kernel_name = "sum_kernel";
	const char * source = R"(
        void __kernel
        sum_kernel(__global float * gC, __global float * gA, __global float * gB)
        {
            int id = get_global_id(0);
            gC[id] = gA[id]+gB[id];
        }
    )";
	// Create the program, build the executable, and extract the entry point
	// for the kernel.
	cl_program program = clCreateProgramWithSource(af_context, 1, &source, NULL, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);
	cl_kernel sumkernel = clCreateKernel(program, kernel_name, &status);
	// Set arguments and launch your kernels
	clSetKernelArg(sumkernel, 0, sizeof(cl_mem), d_C);
	clSetKernelArg(sumkernel, 1, sizeof(cl_mem), d_A);
	clSetKernelArg(sumkernel, 2, sizeof(cl_mem), d_B);
	clEnqueueNDRangeKernel(af_queue, sumkernel, 1, NULL, &order, NULL, 0, NULL, NULL);
	
	// 5. Return control of af::array memory to ArrayFire
	af_unlock_array(dA);
	af_unlock_array(dB);
	af_unlock_array(dC);

	//copy results to output argument
	af_copy_array(out, dC);

	// ... resume ArrayFire operations
	// Because the device pointers, d_x and d_y, were returned to ArrayFire's
	// control by the unlock function, there is no need to free them using
	// clReleaseMemObject()
}

void AFire::printh(float *A,size_t length)
{

	for (int i = 0; i < length; i++)
	{
		std::cout << A[i] << std::endl;
	}
	std::cout << std::endl;
}

void AFire::mulaf(af_array* dC, const af_array dA,
	const af_array dB) {
	//Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//------------------------------------
	//detalles importantes sobre el kernel
	//------------------------------------
	/*suponiendo que las matrices son compatibles para la multiplicación:
	El detalle aquí es el uso adecuado del Kernel en matrixMul.cl
	el kernel está escrito para multiplicar matrices cuyos datos están ordenados por filas,
	sin embargo las matrices de tipo af::array son ordenadas por columnas, si establecemos los argumentos
	en el kernel sin tomar en cuenta este detalle, el resultado será erroneo.
	el Kernel acepta ocho argumentos: suponiendo C=A*B
	1.vector que guarda los datos de C
	2.vector que guarda los datos de A
	3.vector que guarda los datos de B
	4.cero con el tamaño necesario de memoria para un grupo en A
	5.cero con el tamaño necesario de memoria para un grupo en B
	6.ancho de A
	7.ancho de B
	8.alto de A

	sean 2 matrices del tipo af::array

 A=
 0  4   8  12  16  20  24  28
 1  5   9  13  17  21  25  29
 2  6  10  14  18  22  26  30
 3  7  11  15  19  23  27  31

B=
 1   9  17  25  33  41
 2  10  18  26  34  42
 3  11  19  27  35  43
 4  12  20  28  36  44
 5  13  21  29  37  45
 6  14  22  30  38  46
 7  15  23  31  39  47
 8  16  24  32  40  48

	si queremos multiplicar estas 2 matrices usando este kernel, los datos serían:
	dA={0,1,2,...,31}
	dB={1,2,3,...,48}
	así son extraídos con: A.device<cl_mem>();
	como podemos comprobar los están ordenados por columnas al construir las matrices correspondientes

	así como estan definidas las matrices las dimensiones serían:

	ancho de A=8
	ancho de B=6
	alto de A=4

	sin embargo si usamos el kernel en matrixMul.cl, con los argumentos así como están, en realidad se
	estarían multiplicando estas matrices:

A=
  0   1   2   3   4   5   6   7
  8   9  10  11  12  13  14  15
 16  17  18  19  20  21  22  23
 24  25  26  27  28  29  30  31

B=
  1   2   3   4   5   6
  7   8   9  10  11  12
 13  14  15  16  17  18
 19  20  21  22  23  24
 25  26  27  28  29  30
 31  32  33  34  35  36
 37  38  39  40  41  42
 43  44  45  46  47  48

 que son diferentes a las originales, ya que como se dijo al comienzo el kernel asume que los valores estan
 ordenados por filas, si se realiza la multiplicación el resultado será erroneo.

 Sin embargo es posible usar el mismo kernel para obtener un resultado válido, y es invirtiendo el orden de
 los argumentos de entrada, invirtiendo también para cada unos sus dimensiones, es decir colocando B de 6x8 como
 primer argumento y A de 8x4 como segundo argumento, como se podrá comprobar el resultado final es el esperado.

 B=
  1   2   3   4   5   6   7   8
  9  10  11  12  13  14  15  16
 17  18  19  20  21  22  23  24
 25  26  27  28  29  30  31  32
 33  34  35  36  37  38  39  40
 41  42  43  44  45  46  47  48

A=
  0   1   2   3
  4   5   6   7
  8   9  10  11
 12  13  14  15
 16  17  18  19
 20  21  22  23
 24  25  26  27
 28  29  30  31

 B*A=
 672   708   744   780
 1568  1668  1768  1868
 2464  2628  2792  2956
 3360  3588  3816  4044
 4256  4548  4840  5132
 5152  5508  5864  6220

 A*B= (A y B originales)
  672  1568  2464  3360  4256  5152
 708  1668  2628  3588  4548  5508
 744  1768  2792  3816  4840  5864
 780  1868  2956  4044  5132  6220

 no es necesario transponer B*A (cambiados) para obtener A*B (originales) ya que el resultado saldrá
 en forma de vector, estos se ordenarán automáticamente por columna una vez vuelvan a ser parte del campo de C
 o matriz resultado.

 */

	dim_t Bdims[AF_MAX_DIMS];
	af_get_dims(&Bdims[0], &Bdims[1], &Bdims[2], &Bdims[3], dB);
	size_t wB = Bdims[1];
	size_t hB = Bdims[0];

	dim_t Adims[AF_MAX_DIMS];
	af_get_dims(&Adims[0], &Adims[1], &Adims[2], &Adims[3], dA);
	size_t hA = Adims[0];

	af_dtype typef;
	af_get_type(&typef, dA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//matriz resultado
	af_array Rs;
	unsigned ndims = 2;
	dim_t dim[] = { Adims[0],Bdims[1] };
	af_randu(&Rs, ndims, dim, typef);


	size_t program_length = strlen(matrixMul_source);
	int status = CL_SUCCESS;

	//obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, msize * Adims[0] * Adims[1],
		NULL, &status);
	af_get_device_ptr((void**)d_A, dA);

	cl_mem *d_B = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, msize * Bdims[0] * Bdims[1],
		NULL, &status);
	af_get_device_ptr((void**)d_B, dB);

	cl_mem *d_C = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_WRITE_ONLY, msize * Adims[0] * Bdims[1],
		NULL, &status);
	af_get_device_ptr((void**)d_C, Rs);

	//creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(
		af_context, 1, (const char **)&matrixMul_source,
		&program_length, &status);

	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "matrixMul";
	else if (typef == f32)
		kernelName = "matrixMul_sp";
	else;

	cl_kernel mulkernel = clCreateKernel(program,
		kernelName, &status);

	// estableciendo los argumentos, ver detalles importantes sobre el kernel (más arriba)
	int i = 0;
	clSetKernelArg(mulkernel, i++, sizeof(cl_mem), d_C);
	clSetKernelArg(mulkernel, i++, sizeof(cl_mem), d_B);
	clSetKernelArg(mulkernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(mulkernel, i++,
		sizeof(double) * BLOCK_SIZE *BLOCK_SIZE, 0);
	clSetKernelArg(mulkernel, i++,
		sizeof(double) * BLOCK_SIZE *BLOCK_SIZE, 0);
	clSetKernelArg(mulkernel, i++, sizeof(cl_int), &hB);
	clSetKernelArg(mulkernel, i++, sizeof(cl_int), &hA);
	clSetKernelArg(mulkernel, i++, sizeof(cl_int), &wB);

	size_t localWorkSize[] = { BLOCK_SIZE, BLOCK_SIZE };
	size_t globalWorkSize[] = { shrRoundUp(BLOCK_SIZE, hA),
		shrRoundUp(BLOCK_SIZE, wB) };

	//ejecutando el Kernel
	clEnqueueNDRangeKernel(af_queue, mulkernel, 2, 0,
		globalWorkSize, localWorkSize, 0, NULL, NULL);

	//devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(dA);
	af_unlock_array(dB);
	af_unlock_array(Rs);

	// ... reanudando las operaciones ArrayFire
//copiando el resultado en el argumento de salida
	af_copy_array(dC, Rs);
	af_release_array(Rs);
}

char * AFire::find_kernel_test(const char* file)
{
	char * cl_files = AFire::get_env_path(CL_VAR_ENT);
	std::string cd_files_path = cl_files;
	std::string file_name = file;
	std::string path_to_file = cd_files_path + "/" + file_name;
	size_t program_length;
	char *header = oclLoadProgSource(path_to_file.c_str(), "", &program_length);
	return header;
}

char * AFire::helpk(const char * string)
{
	char * str = (char *)malloc(10 * sizeof(char));

	strcpy_s(str, sizeof(char) * 10, "hola ");
	strcat_s(str, sizeof(char) * 10, "will");
	return str;
	
	//return (char*)string;
}

//Obtiene la ruta explicita de una variable de
//entorno, si no se encuentra la variable 
//devolverá "./"
char * AFire::get_env_path(const char * env)
{
	std::string env_s = env;
	env_s += "=";
	int cont = 0;
	bool bool_s = 0;
	size_t pos_s;
	std::string path_s;

	extern char ** environ;
	while (environ[cont] != NULL && !bool_s)
	{
		path_s = std::string(environ[cont]);
		pos_s = path_s.find(env_s);
		if (pos_s != std::string::npos)
		{
			path_s.replace(pos_s, env_s.length(), "");
			bool_s = 1;
		}
		cont += 1;
	}

	if (bool_s)
	{
		char* file_path = (char*)malloc(path_s.length() + 1);
		strcpy_s(file_path, path_s.length() + 1, path_s.c_str());
		return file_path;
	}
	else 
	{
		return "./";
	}
}

/*Obtiene el código fuente de un kernel (OpenCL) y lo concatena 
con el correspondiente encabezado, el programa estará listo para
construirse con el código devuelto
*/
char * AFire::kernel_src(const char* hfile, const char * clfile)
{
	size_t program_length;
	//obteniendo el valor de la variable de entorno
	char * cl_files = AFire::get_env_path(CL_VAR_ENT);
	std::string cl_files_path = cl_files;

	//encabezado comun a los Kernels
	std::string file_name = hfile;
	std::string header_path = cl_files_path + "/" + file_name;
	char *header = oclLoadProgSource(header_path.c_str(), "", &program_length);

	//kernel
	file_name = clfile;
	std::string source_path = cl_files_path + "/" + file_name;
	char * source = oclLoadProgSource(source_path.c_str(), header, &program_length);
	return source;
}

void AFire::sparse_mat_vec_mul(af_array* dC, af_array elmA,
	af_array colA, af_array rowA, af_array dB) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//creando copia de dB
	af_array c;
	af_copy_array(&c, dB);

	//2. Obteniendo parámetros necesarios

	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmA);
	cl_int size_elmA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], colA);
	cl_int size_colA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], rowA);
	cl_int size_rowA = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, elmA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_elmA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, msize*size_elmA,
		NULL, &status);
	af_get_device_ptr((void**)d_elmA, elmA);

	cl_mem *d_colA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_colA,
		NULL, &status);
	af_get_device_ptr((void**)d_colA, colA);

	cl_mem *d_rowA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_rowA,
		NULL, &status);
	af_get_device_ptr((void**)d_rowA, rowA);

	cl_mem *d_b = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, msize*size_colA,
		NULL, &status);
	af_get_device_ptr((void**)d_b, dB);

	cl_mem *d_c = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_colA,
		NULL, &status);
	af_get_device_ptr((void**)d_c, c);

	size_t program_length = strlen(gc_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context,
		1, (const char **)&gc_source, &program_length,
		&status);
	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "sparse_mat_vec_mul";
	else if (typef == f32)
		kernelName = "sparse_mat_vec_mul_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	cl_int step = 0;
	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_colA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_rowA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_b);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_c);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_colA);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &step);

	//6. ejecutando el kernel

	//transpose(L)*b
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//L*b
	step++;
	clSetKernelArg(kernel, 7, sizeof(cl_int), &step);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//D*b
	step++;
	clSetKernelArg(kernel, 7, sizeof(cl_int), &step);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(elmA);
	af_unlock_array(colA);
	af_unlock_array(rowA);
	af_unlock_array(dB);
	af_unlock_array(c);

	//copiando al argumento de salida
	af_copy_array(dC, c);

	af_release_array(c);
}
//--------------
//Algebra Lineal
//--------------

void AFire::SELgj_f(af_array* dC, af_array dA, af_array dB) {
	//Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//acoplando A con B
	af_array Ac;
	af_join(&Ac, 1, dA, dB);

	//ya que el kernel asume la matriz como ordenada por filas
	//debemos transponer la matriz af::array A para obtener la
	//solución correcta, ya que ArrayFire ordena lo elementos
	//por columna
	af_array At;
	af_transpose(&At, Ac, false);

	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2], &_order[3], dA);
	size_t order = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, dA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize * order*(order + 1),
		NULL, &status);
	af_get_device_ptr((void**)d_A, At);

	size_t program_length = strlen(GJordan_source);


	//creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&GJordan_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);
	
	char* kernelName;
	if (typef == f64)
		kernelName = "Gauss_Jordan_f";
	else if (typef == f32)
		kernelName = "Gauss_Jordan_f_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	for (int j = 0; j < order; j++)
	{
		clSetKernelArg(kernel, 2, sizeof(cl_int), &j);
		//ejecutando el Kernel
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0, &globalWorkSize, &localWorkSize,
			0, NULL, NULL);
	}

	//devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(At);


	//hasta aqui At contiene en su última fila
	//y en su diagonal principal, los valores 
	//finales que deben dividirse para obtener
	//la solución final

	//extrayendo la última fila
	af_release_array(Ac);
	af_index_t* indexers = 0;
	af_create_indexers(&indexers);
	af_set_seq_param_indexer(indexers, order, order, 1,
		0, false);
	af_set_seq_param_indexer(indexers, 0, order - 1, 1,
		1, false);
	af_index_gen(&Ac, At, 2, indexers);

	//transponiendo
	af_array Atr;
	af_transpose(&Atr, Ac, false);

	//extrayendo la diagonal
	af_release_array(Ac);
	af_diag_extract(&Ac, At, 0);

	//dividiendo
	af_release_array(At);
	af_div(&At, Atr, Ac, false);

	// copiando el resultado en dC
	af_copy_array(dC, At);

	af_release_array(Atr);
	af_release_array(Ac);
	af_release_array(At);
	af_release_indexers(indexers);
}

void AFire::SELgj_c(af_array* dC, af_array dA, af_array dB) {
	//Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//acoplando A con B
	af_array Ac;
	af_join(&Ac, 1, dA, dB);

	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2], &_order[3], dA);
	size_t order = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, dA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize * order*(order + 1),
		NULL, &status);
	af_get_device_ptr((void**)d_A, Ac);

	size_t program_length = strlen(GJordan_source);


	//creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&GJordan_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "Gauss_Jordan_c";
	else if (typef == f32)
		kernelName = "Gauss_Jordan_c_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	for (int j = 0; j < order; j++)
	{
		clSetKernelArg(kernel, 2, sizeof(cl_int), &j);
		//ejecutando el Kernel
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(Ac);

	//hasta aqui Ac contiene en su última columna
	//y en su diagonal principal, los valores 
	//finales que deben dividirse para obtener
	//la solución final

	//extrayendo la última columna
	af_array Au;
	af_index_t* indexers = 0;
	af_create_indexers(&indexers);
	af_set_seq_param_indexer(indexers, 0, order - 1, 1,
		0, false);
	af_set_seq_param_indexer(indexers, order, order, 1,
		1, false);
	af_index_gen(&Au, Ac, 2, indexers);

	//extrayendo la diagonal
	af_array Ad;
	af_diag_extract(&Ad, Ac, 0);

	//dividiendo
	af_release_array(Ac);
	af_div(&Ac, Au, Ad, false);

	// copiando el resultado en dC
	af_copy_array(dC, Ac);

	af_release_array(Au);
	af_release_array(Ad);
	af_release_array(Ac);
	af_release_indexers(indexers);
}

void AFire::SELgj_fshr(af_array* dC, af_array dA,
	af_array dB) {
	//Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	/*creando copia de B*/
	af_array Bc;
	af_copy_array(&Bc, dB);

	/*ya que el kernel asume la matriz como ordenada por filas
	debemos transponer la matriz af::array A para obtener la
	solución correcta, ya que ArrayFire ordena lo elementos
	por columna*/
	af_array At;
	af_transpose(&At, dA, false);

	//para almacenar el resultado
	af_array Rs;
	af_copy_array(&Rs, dB);

	dim_t Adims[AF_MAX_DIMS];
	af_get_dims(&Adims[0], &Adims[1], &Adims[2], &Adims[3],
		dA);
	size_t order = Adims[0];

	af_dtype typef;
	af_get_type(&typef, dA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	size_t program_length = strlen(GJordan_source);
	int status = CL_SUCCESS;

	//obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize * order * order,
		NULL, &status);
	af_get_device_ptr((void**)d_A, At);

	cl_mem *d_B = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize * order, NULL, &status);
	af_get_device_ptr((void**)d_B, Bc);

	cl_mem *d_C = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_WRITE_ONLY, msize * order, NULL, &status);
	af_get_device_ptr((void**)d_C, Rs);

	//creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&GJordan_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);
	
	char* kernelName;
	if (typef == f64)
		kernelName = "Gauss_Jordan_fshr";
	else if (typef == f32)
		kernelName = "Gauss_Jordan_fshr_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_C);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_B);
	clSetKernelArg(kernel, i++, sizeof(double)*BLOCK_SIZE*BLOCK_SIZE, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	size_t localWorkSize[] = { BLOCK_SIZE,BLOCK_SIZE };
	size_t globalWorkSize[] =
	{ shrRoundUp(localWorkSize[0],order),
		shrRoundUp(localWorkSize[1], order) };

	//ejecutando el Kernel
	clEnqueueNDRangeKernel(af_queue, kernel, 2, 0,
		globalWorkSize, localWorkSize,0, NULL, NULL);

	//devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(At);
	af_unlock_array(Bc);
	af_unlock_array(Rs);

	//copiando el resultado
	af_copy_array(dC, Rs);

	af_release_array(At);
	af_release_array(Bc);
	af_release_array(Rs);
}

void AFire::SELgj_f2d(af_array* dC, af_array dA,
	af_array dB) {
	//Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	/*creando copia de B*/
	af_array Bc;
	af_copy_array(&Bc, dB);

	//para almacenar el resultado
	af_array Rs;
	af_copy_array(&Rs, dB);

	/*ya que el kernel asume la matriz como ordenada por filas
	debemos transponer la matriz af::array A para obtener la
	solución correcta, ya que ArrayFire ordena lo elementos
	por columna*/
	af_array At;
	af_transpose(&At, dA, false); 

	dim_t Adims[AF_MAX_DIMS];
	af_get_dims(&Adims[0], &Adims[1], &Adims[2], &Adims[3],
		dA);
	size_t order = Adims[0];

	af_dtype typef;
	af_get_type(&typef, dA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	size_t program_length = strlen(GJordan_source);
	int status = CL_SUCCESS;

	//obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize * order * order,
		NULL, &status);
	af_get_device_ptr((void**)d_A, At);

	cl_mem *d_B = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize * order, NULL, &status);
	af_get_device_ptr((void**)d_B, Bc);

	cl_mem *d_C = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_WRITE_ONLY, msize * order, NULL, &status);
	af_get_device_ptr((void**)d_C, Rs);

	//creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&GJordan_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);
	
	char* kernelName;
	if (typef == f64)
		kernelName = "Gauss_Jordan_f2d";
	else if (typef == f32)
		kernelName = "Gauss_Jordan_f2d_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_C);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_B);
    clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	size_t localWorkSize[] = { BLOCK_SIZE,BLOCK_SIZE };
	size_t globalWorkSize[] = { BLOCK_SIZE,BLOCK_SIZE };
	
	//ejecutando el Kernel
	clEnqueueNDRangeKernel(af_queue, kernel, 2, 0,
		globalWorkSize, localWorkSize, 0, NULL, NULL);

	//devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(At);
	af_unlock_array(Bc);
	af_unlock_array(Rs);

	//copiando el resultado
	af_copy_array(dC, Rs);

	af_release_array(At);
	af_release_array(Bc);
	af_release_array(Rs);
}

void AFire::SELgj_c2d(af_array* dC, af_array dA,
	af_array dB) {
	//Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	/*creando copia de B*/
	af_array Bc;
	af_copy_array(&Bc, dB);

	//para almacenar el resultado
	af_array Rs;
	af_copy_array(&Rs, dB);

	/*creando copia de A*/
	af_array At;
	af_copy_array(&At, dA);

	dim_t Adims[AF_MAX_DIMS];
	af_get_dims(&Adims[0], &Adims[1], &Adims[2], &Adims[3],
		dA);
	size_t order = Adims[0];

	af_dtype typef;
	af_get_type(&typef, dA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	size_t program_length = strlen(GJordan_source);
	int status = CL_SUCCESS;

	//obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize * order * order,
		NULL, &status);
	af_get_device_ptr((void**)d_A, At);

	cl_mem *d_B = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize * order, NULL, &status);
	af_get_device_ptr((void**)d_B, Bc);

	cl_mem *d_C = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_WRITE_ONLY, msize * order, NULL, &status);
	af_get_device_ptr((void**)d_C, Rs);

	//creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&GJordan_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);
	
	char* kernelName;
	if (typef == f64)
		kernelName = "Gauss_Jordan_c2d";
	else if (typef == f32)
		kernelName = "Gauss_Jordan_c2d_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_C);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_B);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	size_t localWorkSize[] = { BLOCK_SIZE,BLOCK_SIZE };
	size_t globalWorkSize[] = { BLOCK_SIZE,BLOCK_SIZE };

	//ejecutando el Kernel
	clEnqueueNDRangeKernel(af_queue, kernel, 2, 0,
		globalWorkSize, localWorkSize, 0, NULL, NULL);

	//devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(At);
	af_unlock_array(Bc);
	af_unlock_array(Rs);

	//copiando el resultado
	af_copy_array(dC, Rs);

	af_release_array(At);
	af_release_array(Bc);
	af_release_array(Rs);
}

void AFire::prueba_shr(af_array* dC, af_array dA, 
	af_array dB) {
	//Obteniendo el dispositivo, contexto y la cola usada
	//por ArrayFire
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();
	
	dim_t Adims[AF_MAX_DIMS];
	af_get_dims(&Adims[0], &Adims[1], &Adims[2], &Adims[3], dA);
	size_t order = Adims[0];

	af_dtype typef;
	af_get_type(&typef, dA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	size_t program_length = strlen(GJordan_source);
	int status = CL_SUCCESS;

	//creando copia de dA y dB, para poder modificarlos
	af_array Bc;
	af_copy_array(&Bc, dB);
	af_array Ac;
	af_copy_array(&Ac, dA);

	//para almacenar el resultado
	af_array Rs;
	af_copy_array(&Rs, dB);

	//obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize * order * order,
		NULL, &status);
	af_get_device_ptr((void**)d_A, Ac);

	cl_mem *d_B = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize * order, NULL, &status);
	af_get_device_ptr((void**)d_B, Bc);

	cl_mem *d_C = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_WRITE_ONLY, msize * order, NULL, &status);
	af_get_device_ptr((void**)d_C, Rs);

	
	//creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&GJordan_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);
	
	char* kernelName;
	if (typef == f64)
		kernelName = "prueba_shr";
	else if (typef == f32)
		kernelName = "prueba_shr_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_C);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_B);
	clSetKernelArg(kernel, i++, sizeof(double)*BLOCK_SIZE*BLOCK_SIZE, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	size_t localWorkSize[] = { BLOCK_SIZE, BLOCK_SIZE };
	size_t globalWorkSize[] =
		{ shrRoundUp(localWorkSize[0], order),shrRoundUp(localWorkSize[1], order) };
	//ejecutando el Kernel
	clEnqueueNDRangeKernel(af_queue, kernel, 2, 0,
		globalWorkSize, localWorkSize, 0, NULL, NULL);

	//devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(Ac);
	af_unlock_array(Bc);
	af_unlock_array(Rs);

	//copiando el resultado
	af_copy_array(dC, Rs);

	af_release_array(Ac);
	af_release_array(Bc);
	af_release_array(Rs);
}

void AFire::SEL_gc(af_array* C, af_array A, af_array b,
	double Ierr) {

	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], A);
	size_t order = _order[0];

	af_dtype typef;
	af_get_type(&typef, A);

	double normr;
	//af_array de ayuda
	af_array zero;
	dim_t d_order[] = { 1 };
	af_constant(&zero, 0, 1, d_order, typef);
	af_array Ax0;
	af_array rtxp;
	af_array ptxz;
	af_array axp;
	af_array B;
	af_array axz;
	af_array copyr;
	af_array rtxz; 
	af_array rsp;
	af_array Bxp;

	//x0=b
	af_array x0;
	af_copy_array(&x0, b);

	//r=b-A*x0
	af_array r;
	af_matmul(&Ax0, A, x0, AF_MAT_NONE, AF_MAT_NONE);
	af_sub(&r, b, Ax0, false);

	//p = r
	af_array p;
	af_copy_array(&p, r);

	//z=A*p
	af_array z;
	af_matmul(&z, A, p, AF_MAT_NONE, AF_MAT_NONE);

	//a = (r'*p)/(p'*z)
	af_array a;
	af_matmul(&rtxp, r, p, AF_MAT_TRANS, AF_MAT_NONE);
	af_matmul(&ptxz, p, z, AF_MAT_TRANS, AF_MAT_NONE);
	af_div(&a, rtxp, ptxz, false);

	//x = x0 + a.*p
	af_array x;
	af_mul(&axp, a, p, true);
	af_add(&x, x0, axp, false);

	int contin;
	int I = 0;
	for (int i = 0; i < order; i++)
	{
		//r -= a.*z
		af_copy_array(&copyr, r);
		af_mul(&axz, a, z, true);
		af_sub(&r, copyr, axz, false);

		af_norm(&normr, r, AF_NORM_EUCLID, 1, 1);
		if (normr <= Ierr)
			break;

		//B = -(r'*z)/(p'*z)
		af_matmul(&rtxz, r, z, AF_MAT_TRANS, AF_MAT_NONE);
		af_matmul(&ptxz, p, z, AF_MAT_TRANS, AF_MAT_NONE);
		af_div(&rsp, rtxz, ptxz, false);
		af_sub(&B, zero, rsp, false);
	
		//p = r + B.*p
		af_mul(&Bxp, B, p, true);
		af_add(&p, r, Bxp, false);
		
		//z = A*p
		af_matmul(&z, A, p, AF_MAT_NONE, AF_MAT_NONE);
		
		//a = (r'*p)/(p'*z)
		af_matmul(&rtxp, r, p, AF_MAT_TRANS, AF_MAT_NONE);
		af_matmul(&ptxz, p, z, AF_MAT_TRANS, AF_MAT_NONE);
		af_div(&a, rtxp, ptxz, false);
		
		//x += a.*p
		af_mul(&axp, a, p, true);
		af_copy_array(&x0, x);
		af_add(&x, x0, axp, false);

		I++;
	}

	//copiando el resultado en el argumento de salida
	af_copy_array(C, x);

	//liberando objetos af_array usados
	af_release_array(zero);
	af_release_array(Ax0);
	af_release_array(rtxp);
	af_release_array(ptxz);
	af_release_array(axp);
	af_release_array(B);
	af_release_array(axz);
	af_release_array(copyr);
	af_release_array(rtxz);
	af_release_array(rsp);
	af_release_array(Bxp);
	af_release_array(x0);
	af_release_array(r);
	af_release_array(p);
	af_release_array(z);
	af_release_array(a);
	af_release_array(x);
}

void AFire::SELgc_sparse(af_array* C, af_array elmA,
	af_array colA, af_array rowA, af_array b,
	double Ierr) {

	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], colA);
	size_t order = _order[0];

	af_dtype typef;
	af_get_type(&typef, elmA);

	double normr;
	//af_array de ayuda
	af_array zero;
	dim_t d_order[] = { 1 };
	af_constant(&zero, 0, 1, d_order, typef);
	af_array Ax0;
	af_array rtxp;
	af_array ptxz;
	af_array axp;
	af_array B;
	af_array axz;
	af_array copyr;
	af_array rtxz;
	af_array rsp;
	af_array Bxp;

	//x0=b
	af_array x0;
	af_copy_array(&x0, b);

	//r=b-A*x0
	af_array r;
	//af_matmul(&Ax0, A, x0, AF_MAT_NONE, AF_MAT_NONE);
	AFire::sparse_mat_vec_mul(&Ax0, elmA, colA, rowA, x0);
	af_sub(&r, b, Ax0, false);

	//p = r
	af_array p;
	af_copy_array(&p, r);

	//z=A*p
	af_array z;
	//af_matmul(&z, A, p, AF_MAT_NONE, AF_MAT_NONE);
	AFire::sparse_mat_vec_mul(&z, elmA, colA, rowA, p);

	//a = (r'*p)/(p'*z)
	af_array a;
	af_matmul(&rtxp, r, p, AF_MAT_TRANS, AF_MAT_NONE);
	af_matmul(&ptxz, p, z, AF_MAT_TRANS, AF_MAT_NONE);
	af_div(&a, rtxp, ptxz, false);

	//x = x0 + a.*p
	af_array x;
	af_mul(&axp, a, p, true);
	af_add(&x, x0, axp, false);

	int contin;
	int I = 0;
	for (int i = 0; i < order; i++)
	{
		//r -= a.*z
		af_copy_array(&copyr, r);
		af_mul(&axz, a, z, true);
		af_sub(&r, copyr, axz, false);

		af_norm(&normr, r, AF_NORM_EUCLID, 1, 1);
		if (normr <= Ierr)
			break;

		//B = -(r'*z)/(p'*z)
		af_matmul(&rtxz, r, z, AF_MAT_TRANS, AF_MAT_NONE);
		af_matmul(&ptxz, p, z, AF_MAT_TRANS, AF_MAT_NONE);
		af_div(&rsp, rtxz, ptxz, false);
		af_sub(&B, zero, rsp, false);

		//p = r + B.*p
		af_mul(&Bxp, B, p, true);
		af_add(&p, r, Bxp, false);

		//z = A*p
		//af_matmul(&z, A, p, AF_MAT_NONE, AF_MAT_NONE);
		AFire::sparse_mat_vec_mul(&z, elmA, colA, rowA, p);

		//a = (r'*p)/(p'*z)
		af_matmul(&rtxp, r, p, AF_MAT_TRANS, AF_MAT_NONE);
		af_matmul(&ptxz, p, z, AF_MAT_TRANS, AF_MAT_NONE);
		af_div(&a, rtxp, ptxz, false);

		//x += a.*p
		af_mul(&axp, a, p, true);
		af_copy_array(&x0, x);
		af_add(&x, x0, axp, false);

		I++;
	}

	//copiando el resultado en el argumento de salida
	af_copy_array(C, x);

	//liberando objetos af_array usados
	af_release_array(zero);
	af_release_array(Ax0);
	af_release_array(rtxp);
	af_release_array(ptxz);
	af_release_array(axp);
	af_release_array(B);
	af_release_array(axz);
	af_release_array(copyr);
	af_release_array(rtxz);
	af_release_array(rsp);
	af_release_array(Bxp);
	af_release_array(x0);
	af_release_array(r);
	af_release_array(p);
	af_release_array(z);
	af_release_array(a);
	af_release_array(x);
}

array AFire::SEL_gc(array A, array b, double Ierr){

	bool gfor_stat = gforGet();
	gforSet(true);
	size_t order = A.dims(0);

	array x0 = b;
	array r = b - matmul(A,x0);
	array p = r;
	array z = matmul(A, p);
	array a = matmul(r, p, AF_MAT_TRANS) /
		matmul(p, z, AF_MAT_TRANS);
	array x = x0 + a * p;
	array B;

	int I = 0;
	for (int i = 0; i < order; i++)
	{
		r -= a * z;
		if (norm(r) <= Ierr)
			break;
		B = -matmul(r, z, AF_MAT_TRANS) /
			matmul(p, z, AF_MAT_TRANS);
		p = r + B * p;
		z = matmul(A, p);
		a = matmul(r, p, AF_MAT_TRANS) /
			matmul(p, z, AF_MAT_TRANS);
		x += a * p;
		I++;
	}
	gforSet(gfor_stat);
	return x;
}

void AFire::SELchol_c(af_array* dC, af_array dA, af_array dB) {
	//Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//copia de A y B
	af_array Ac;
	af_array Bc;
	af_copy_array(&Ac, dA);
	af_copy_array(&Bc, dB);

	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2], &_order[3], dA);
	size_t order = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, dA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;
	
	//obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*order*order,
		NULL, &status);
	af_get_device_ptr((void**)d_A, Ac);

	cl_mem *d_B = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*order,
		NULL, &status);
	af_get_device_ptr((void**)d_B, Bc);

	size_t program_length = strlen(Cholesky_source);


	//creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&Cholesky_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "Cholesky_c";
	else if (typef == f32)
		kernelName = "Cholesky_c_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_B);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	for (int j = 0; j < order; j++)
	{
		//Se modifican los elementos de la columna j a partir 
		//de la fila j + 1.
		//A(n, j) = A(n, j) / sqrt(A(j, j)) n > j
		int sstep = 0;
		clSetKernelArg(kernel, 4, sizeof(cl_int), &j);
		clSetKernelArg(kernel, 5, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);

		//para una columna s, s > j, se modifican las filas n >= s
		//A(n, s) = A(n, s) - A(s, j)*A(n, j) s > j, n >= s
		sstep++;
		clSetKernelArg(kernel, 5, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//cada elemento diagonal es reemplazado por su raiz cuadrada
	//asi termina la factorización, la matriz Ac, tendra en su 
	//parte triangular inferior, los elementos de la factorización
	//de cholesky
	int fstep = 2;
	clSetKernelArg(kernel, 5, sizeof(cl_int), &fstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//luego de la factorización se tiene el sistema equivalente
	//L*transpose(L)x=b que debe resolverse separadamente para
	//Ly=b
	//y luego
	//transpose(L)x=y

	//resolviendo Ly=b
	fstep++;
	clSetKernelArg(kernel, 5, sizeof(cl_int), &fstep);
	for (int j = 0; j < order; j++)
	{
		//para un paso j, modificará el vector b tal que :
		//b[n] = b[n] - b[j] * L[n, j] / L[j, j] n > j
		clSetKernelArg(kernel, 4, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	fstep++;
	clSetKernelArg(kernel, 5, sizeof(cl_int), &fstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//resolviendo transpose(L)x=y
	fstep++;
	clSetKernelArg(kernel, 5, sizeof(cl_int), &fstep);
	for (int j = 0; j < order; j++)
	{
		//para un paso j, modifica b[order-j], este valor
		//es un componente final de la solución del sistema
		clSetKernelArg(kernel, 4, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(Ac);
	af_unlock_array(Bc);

	// copiando el resultado en dC
	af_copy_array(dC, Bc);

	af_release_array(Ac);
	af_release_array(Bc);
}

void AFire::fac_chol_c(af_array* L, af_array A) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//2. Obteniendo parámetros necesarios
	//copia de A
	af_array Ac;
	af_copy_array(&Ac, A);

	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], A);
	size_t order = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, A);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*order*order,
		NULL, &status);
	af_get_device_ptr((void**)d_A, Ac);

	size_t program_length = strlen(Cholesky_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&Cholesky_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "Cholesky_c";
	else if (typef == f32)
		kernelName = "Cholesky_c_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	//6.Ejecutando el kernel
	for (int j = 0; j < order; j++)
	{
		//Se modifican los elementos de la columna j a partir 
		//de la fila j + 1.
		//A(n, j) = A(n, j) / sqrt(A(j, j)) n > j
		int sstep = 0;
		clSetKernelArg(kernel, 4, sizeof(cl_int), &j);
		clSetKernelArg(kernel, 5, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);

		//para una columna s, s > j, se modifican las filas n >= s
		//A(n, s) = A(n, s) - A(s, j)*A(n, j) s > j, n >= s
		sstep++;
		clSetKernelArg(kernel, 5, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//cada elemento diagonal es reemplazado por su raiz cuadrada
	//asi termina la factorización, la matriz Ac, tendra en su 
	//parte triangular inferior, los elementos de la factorización
	//de cholesky
	int fstep = 2;
	clSetKernelArg(kernel, 5, sizeof(cl_int), &fstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(Ac);

	// copiando el resultado en dC
	af_copy_array(L, Ac);

	af_release_array(Ac);
}

void AFire::SELldlt_c(af_array* dC, af_array dA, af_array dB) {
	//Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//copia de A y B
	af_array Ac;
	af_array Bc;
	af_copy_array(&Ac, dA);
	af_copy_array(&Bc, dB);

	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2], &_order[3], dA);
	size_t order = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, dA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*order*order,
		NULL, &status);
	af_get_device_ptr((void**)d_A, Ac);

	cl_mem *d_B = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*order,
		NULL, &status);
	af_get_device_ptr((void**)d_B, Bc);

	size_t program_length = strlen(ldlt_source);


	//creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&ldlt_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "ldlt_c";
	else if (typef == f32)
		kernelName = "ldlt_c_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_B);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	for (int j = 0; j < order; j++)
	{
		//Se modifican los elementos de la columna j a partir 
		//de la fila j + 1.
		//A(n,j)=A(n,j)/A(j,j) n>j
		int sstep = 0;
		clSetKernelArg(kernel, 4, sizeof(cl_int), &j);
		clSetKernelArg(kernel, 5, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);

		//para una columna s, s > j, se modifican las filas 
		//n >= s
	    //A(n, s) = A(n, s) - A(s, j)*A(n, j)*A(j, j) 
		//s > j, n >= s
		sstep++;
		clSetKernelArg(kernel, 5, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//luego de la factorización se tiene el sistema equivalente
	//L*D*transpose(L)x=b que debe resolverse separadamente para
	//Ly=b, luego para Dz=y y finalmente para transpose(L)x=z
	
	//resolviendo Ly=b
	int fstep = 2;
	clSetKernelArg(kernel, 5, sizeof(cl_int), &fstep);
	for (int j = 0; j < order; j++)
	{
		//para un paso j, modificará el vector b tal que :
		//b[n] = b[n] - b[j] * L[n, j] n > j
		clSetKernelArg(kernel, 4, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//resolviendo Dz=y
	fstep++;
	clSetKernelArg(kernel, 5, sizeof(cl_int), &fstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//resolviendo transpose(L)x=z
	fstep++;
	clSetKernelArg(kernel, 5, sizeof(cl_int), &fstep);
	for (int j = 0; j < order; j++)
	{
		//para un paso j modificará el elemento b[i] 
		//tal que i = n - j:
		//b[i] = b[i] - sumatoria{ j = i + 1,n }(z[j] * L[j, i])
		//este valor será la solución final x[i] del 
		//sistema Ax = b
		clSetKernelArg(kernel, 4, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(Ac);
	af_unlock_array(Bc);

	// copiando el resultado en dC
	af_copy_array(dC, Bc);

	af_release_array(Ac);
	af_release_array(Bc);
}

void AFire::fac_ldlt_c(af_array* L, af_array A) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//2. Obteniendo parámetros necesarios
	//copia de A
	af_array Ac;
	af_copy_array(&Ac, A);

	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], A);
	size_t order = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, A);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*order*order,
		NULL, &status);
	af_get_device_ptr((void**)d_A, Ac);

	size_t program_length = strlen(ldlt_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&ldlt_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "ldlt_c";
	else if (typef == f32)
		kernelName = "ldlt_c_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	//6.Ejecutando el kernel
	for (int j = 0; j < order; j++)
	{
		//Se modifican los elementos de la columna j a partir 
		//de la fila j + 1.
		//A(n,j)=A(n,j)/A(j,j) n>j
		int sstep = 0;
		clSetKernelArg(kernel, 4, sizeof(cl_int), &j);
		clSetKernelArg(kernel, 5, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);

		//para una columna s, s > j, se modifican las filas 
		//n >= s
		//A(n, s) = A(n, s) - A(s, j)*A(n, j)*A(j, j) 
		//s > j, n >= s
		sstep++;
		clSetKernelArg(kernel, 5, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(Ac);

	// copiando el resultado en dC
	af_copy_array(L, Ac);

	af_release_array(Ac);
}

void AFire::fac_sparse_chol_c(af_array elmA, af_array colA,
	af_array rowA, af_array elmL, af_array colL,
	af_array rowL) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();
	
	//2. Obteniendo parámetros necesarios

	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmA);
	size_t size_elmA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], colA);
	size_t size_colA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], rowA);
	size_t size_rowA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmL);
	size_t size_elmL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], colL);
	size_t size_colL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], rowL);
	size_t size_rowL = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, elmA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_elmA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, msize*size_elmA,
		NULL, &status);
	af_get_device_ptr((void**)d_elmA, elmA);

	cl_mem *d_colA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_colA,
		NULL, &status);
	af_get_device_ptr((void**)d_colA, colA);

	cl_mem *d_rowA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_rowA,
		NULL, &status);
	af_get_device_ptr((void**)d_rowA, rowA);

	cl_mem *d_elmL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_elmL,
		NULL, &status);
	af_get_device_ptr((void**)d_elmL, elmL);

	cl_mem *d_colL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_colL,
		NULL, &status);
	af_get_device_ptr((void**)d_colL, colL);

	cl_mem *d_rowL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_rowL,
		NULL, &status);
	af_get_device_ptr((void**)d_rowL, rowL);

	size_t program_length = strlen(Cholesky_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&Cholesky_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "Cholesky_sparse_c";
	else if (typef == f32)
		kernelName = "Cholesky_sparse_c_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	int step = 0;
	int sstep = -1;

	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_colA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_rowA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_colL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_rowL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_elmA);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_colA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &step);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &sstep);

	//6. ejecutando el kernel

	//primera ejecución con sstep=-1 copiará los elementos
	//de elmA en el lugar correspondiente de elmL, tomar
	//en cuenta que los elementos iniciales del elmL son 
	//todos cero.
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//6.Ejecutando el kernel
	for (int j = 0; j < size_colA-1; j++)
	{
		//Se modifican los elementos de la columna j a partir 
		//de la fila j + 1.
		//A(n, j) = A(n, j) / sqrt(A(j, j)) n > j
		int sstep = 0;
		clSetKernelArg(kernel, 10, sizeof(cl_int), &j);
		clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);

		//para una columna s, s > j, se modifican las filas n >= s
		//A(n, s) = A(n, s) - A(s, j)*A(n, j) s > j, n >= s
		sstep++;
		clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//cada elemento diagonal es reemplazado por su raiz cuadrada
	//asi termina la factorización, la matriz Ac, tendra en su 
	//parte triangular inferior, los elementos de la factorización
	//de cholesky
	int fstep = 2;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &fstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(elmA);
	af_unlock_array(colA);
	af_unlock_array(rowA);
	af_unlock_array(elmL);
	af_unlock_array(colL);
	af_unlock_array(rowL);
}

void AFire::SELchol_sparse_c(af_array* dC, af_array elmA,
	af_array colA, af_array rowA, af_array elmL,
	af_array colL,af_array rowL, af_array dB) {
	
	AFire::fac_sparse_chol_c(elmA, colA, rowA, elmL,
		colL, rowL);
	AFire::SELchol_sparse_c(dC, elmL, colL, rowL, dB);
	/*//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//creando copia de dB
	af_array b;
	af_copy_array(&b, dB);

	//2. Obteniendo parámetros necesarios

	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmA);
	cl_int size_elmA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], colA);
	cl_int size_colA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], rowA);
	cl_int size_rowA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmL);
	cl_int size_elmL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], colL);
	cl_int size_colL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], rowL);
	cl_int size_rowL = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, elmA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_elmA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, msize*size_elmA,
		NULL, &status);
	af_get_device_ptr((void**)d_elmA, elmA);

	cl_mem *d_colA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_colA,
		NULL, &status);
	af_get_device_ptr((void**)d_colA, colA);

	cl_mem *d_rowA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_rowA,
		NULL, &status);
	af_get_device_ptr((void**)d_rowA, rowA);

	cl_mem *d_elmL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_elmL,
		NULL, &status);
	af_get_device_ptr((void**)d_elmL, elmL);

	cl_mem *d_colL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_colL,
		NULL, &status);
	af_get_device_ptr((void**)d_colL, colL);

	cl_mem *d_rowL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_rowL,
		NULL, &status);
	af_get_device_ptr((void**)d_rowL, rowL);

	cl_mem *d_b = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_colA,
		NULL, &status);
	af_get_device_ptr((void**)d_b, b);

	size_t program_length = strlen(Cholesky_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context,
		1, (const char **)&Cholesky_source, &program_length,
		&status);
	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "Cholesky_sparse_c";
	else if (typef == f32)
		kernelName = "Cholesky_sparse_c_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	cl_int step = 0;
	cl_int sstep = -1;

	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_colA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_rowA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_colL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_rowL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_elmA);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_colA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_b);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &step);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &sstep);

	//6. ejecutando el kernel

	//primera ejecución con sstep=-1 copiará los elementos
	//de elmA en el lugar correspondiente de elmL, tomar
	//en cuenta que los elementos iniciales del elmL son 
	//todos cero.
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	for (cl_int j = 0; j < size_colA - 1; j++)
	{
		//Se modifican los elementos de la columna j a partir 
		//de la fila j + 1.
		//A(n, j) = A(n, j) / sqrt(A(j, j)) n > j
		sstep = 0;
		clSetKernelArg(kernel, 10, sizeof(cl_int), &j);
		clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);

		//para una columna s, s > j, se modifican las filas n >= s
		//A(n, s) = A(n, s) - A(s, j)*A(n, j) s > j, n >= s
		sstep++;
		clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//cada elemento diagonal es reemplazado por su raiz cuadrada
	//asi termina la factorización, la matriz Ac, tendra en su 
	//parte triangular inferior, los elementos de la factorización
	//de cholesky
	sstep++;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//termina la factorización
	//resolviendo el sistema Ly=b (y=transpose(L)*x)
	sstep++;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
	for (cl_int j = 0; j < size_colA - 1; j++) {
		clSetKernelArg(kernel, 10, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}
	sstep++;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);
	
	//resolviendo el sistema transpose(L)*x=y
	sstep++;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);

	for (cl_int j = 0; j < size_colA; j++) {
		clSetKernelArg(kernel, 10, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(elmA);
	af_unlock_array(colA);
	af_unlock_array(rowA);
	af_unlock_array(elmL);
	af_unlock_array(colL);
	af_unlock_array(rowL);
	af_unlock_array(b);

	//copiando al argumento de salida
	af_copy_array(dC, b);

	af_release_array(b);*/
}

void AFire::SELchol_sparse_c(af_array* dC, af_array elmL,
	af_array colL, af_array rowL, af_array dB) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//creando copia de dB
	af_array b;
	af_copy_array(&b, dB);

	//2. Obteniendo parámetros necesarios

	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmL);
	size_t size_elmL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], colL);
	size_t size_colL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], rowL);
	size_t size_rowL = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, elmL);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_elmL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_elmL,
		NULL, &status);
	af_get_device_ptr((void**)d_elmL, elmL);

	cl_mem *d_colL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_colL,
		NULL, &status);
	af_get_device_ptr((void**)d_colL, colL);

	cl_mem *d_rowL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_rowL,
		NULL, &status);
	af_get_device_ptr((void**)d_rowL, rowL);

	cl_mem *d_b = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_colL,
		NULL, &status);
	af_get_device_ptr((void**)d_b, b);

	size_t program_length = strlen(Cholesky_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context,
		1, (const char **)&Cholesky_source, &program_length,
		&status);
	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "Cholesky_sparse_c";
	else if (typef == f32)
		kernelName = "Cholesky_sparse_c_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	int step = 0;
	int sstep = 0;
	int kk = 0;
	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_colL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_rowL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &kk);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_colL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_b);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &step);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &sstep);

	//6. ejecutando el kernel

	//resolviendo el sistema Ly=b (y=transpose(L)*x)
	sstep=3;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
	for (int j = 0; j < size_colL - 1; j++) {

		clSetKernelArg(kernel, 10, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}
	sstep++;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//resolviendo el sistema transpose(L)*x=y
	sstep++;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);

	for (int j = 0; j < size_colL; j++) {

		clSetKernelArg(kernel, 10, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(elmL);
	af_unlock_array(colL);
	af_unlock_array(rowL);
	af_unlock_array(b);

	//copiando al argumento de salida
	af_copy_array(dC, b);

	af_release_array(b);
}

void AFire::fac_sparse_chol_sks(af_array elmA,
	af_array idxA) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//2. Obteniendo parámetros necesarios
	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmA);
	size_t size_elmL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], idxA);
	size_t size_idxL = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, elmA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_elmL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_elmL,
		NULL, &status);
	af_get_device_ptr((void**)d_elmL, elmA);

	cl_mem *d_idxL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_idxL,
		NULL, &status);
	af_get_device_ptr((void**)d_idxL, idxA);

	size_t program_length = strlen(Cholesky_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context,
		1, (const char **)&Cholesky_source, &program_length,
		&status);
	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "chol_sparse_sks";
	else if (typef == f32)
		kernelName = "chol_sparse_sks_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_idxL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_idxL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);

	cl_int sstep = 0;
	//6.Ejecutando el kernel
	for (cl_int j = 0; j < size_idxL - 1; j++)
	{
		//Se modifican los elementos de la columna j a partir 
		//de la fila j + 1.
		//A(n, j) = A(n, j) / sqrt(A(j, j)) n > j
		sstep = 0;
		clSetKernelArg(kernel, 6, sizeof(cl_int), &j);
		clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);

		//para una columna s, s > j, se modifican las filas n >= s
		//A(n,s)=A(n,s)-A(s,j)*A(n,j) s>j, n>=s
		sstep++;
		clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}
	sstep++;
	clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(elmA);
	af_unlock_array(idxA);
}

void AFire::SELchol_sparse_sks(af_array* dC, af_array elmL,
	af_array idxL, af_array dB) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//creando copia de dB
	af_array b;
	af_copy_array(&b, dB);

	//2. Obteniendo parámetros necesarios

	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmL);
	size_t size_elmL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], idxL);
	size_t size_idxL = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, elmL);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_elmL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, msize*size_elmL,
		NULL, &status);
	af_get_device_ptr((void**)d_elmL, elmL);

	cl_mem *d_idxL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_idxL,
		NULL, &status);
	af_get_device_ptr((void**)d_idxL, idxL);

	cl_mem d_b = (cl_mem)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_idxL,
		NULL, &status);
	af_get_device_ptr((void**)d_b, b);

	size_t program_length = strlen(Cholesky_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context,
		1, (const char **)&Cholesky_source, &program_length,
		&status);
	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "chol_sparse_sks";
	else if (typef == f32)
		kernelName = "chol_sparse_sks_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	cl_int kk = 0;
	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_idxL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &kk);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_idxL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_b);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);

	//6. ejecutando el kernel
	//resolviendo el sistema Ly=b (y=transpose(L)*x)
	cl_int sstep = 3;
	clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
	for (cl_int j = 0; j < size_idxL - 1; j++) {

		clSetKernelArg(kernel, 6, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}
	sstep++;
	clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//resolviendo el sistema transpose(L)*x=y
	sstep++;
	clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
	for (cl_int j = 0; j < size_idxL; j++) {
		clSetKernelArg(kernel, 6, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(elmL);
	af_unlock_array(idxL);
	af_unlock_array(b);

	//copiando al argumento de salida
	af_copy_array(dC, b);

	af_release_array(b);
}

void AFire::fac_sparse_ldlt_c(af_array elmA, af_array colA,
	af_array rowA, af_array elmL, af_array colL,
	af_array rowL) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//2. Obteniendo parámetros necesarios

	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmA);
	size_t size_elmA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], colA);
	size_t size_colA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], rowA);
	size_t size_rowA = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmL);
	size_t size_elmL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], colL);
	size_t size_colL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], rowL);
	size_t size_rowL = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, elmA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_elmA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, msize*size_elmA,
		NULL, &status);
	af_get_device_ptr((void**)d_elmA, elmA);

	cl_mem *d_colA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_colA,
		NULL, &status);
	af_get_device_ptr((void**)d_colA, colA);

	cl_mem *d_rowA = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_rowA,
		NULL, &status);
	af_get_device_ptr((void**)d_rowA, rowA);

	cl_mem *d_elmL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_elmL,
		NULL, &status);
	af_get_device_ptr((void**)d_elmL, elmL);

	cl_mem *d_colL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_colL,
		NULL, &status);
	af_get_device_ptr((void**)d_colL, colL);

	cl_mem *d_rowL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_rowL,
		NULL, &status);
	af_get_device_ptr((void**)d_rowL, rowL);

	size_t program_length = strlen(ldlt_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context,
		1, (const char **)&ldlt_source, &program_length,
		&status);
	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "ldlt_sparse_c";
	else if (typef == f32)
		kernelName = "ldlt_sparse_c_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	int step = 0;
	int sstep = -1;

	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_colA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_rowA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_colL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_rowL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_elmA);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_colA);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &step);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &sstep);

	//6. ejecutando el kernel

	//primera ejecución con sstep=-1 copiará los elementos
	//de elmA en el lugar correspondiente de elmL, tomar
	//en cuenta que los elementos iniciales del elmL son 
	//todos cero.
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//6.Ejecutando el kernel
	for (int j = 0; j < size_colA - 1; j++)
	{
		//Se modifican los elementos de la columna j a partir 
		//de la fila j + 1.
		//A(n,j)=A(n,j)/A(j,j) n>j
		int sstep = 0;
		clSetKernelArg(kernel, 10, sizeof(cl_int), &j);
		clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);

		//para una columna s, s > j, se modifican las filas n >= s
		//A(n,s)=A(n,s)-A(s,j)*A(n,j)*A(j,j) s>j, n>=s
		sstep++;
		clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(elmA);
	af_unlock_array(colA);
	af_unlock_array(rowA);
	af_unlock_array(elmL);
	af_unlock_array(colL);
	af_unlock_array(rowL);
}

void AFire::SELldlt_sparse_c(af_array* dC, af_array elmA,
	af_array colA, af_array rowA, af_array elmL,
	af_array colL, af_array rowL, af_array dB) {

	AFire::fac_sparse_ldlt_c(elmA, colA, rowA, elmL,
		colL, rowL);
	AFire::SELldlt_sparse_c(dC, elmL, colL, rowL, dB);
}

void AFire::SELldlt_sparse_c(af_array* dC, af_array elmL,
	af_array colL, af_array rowL, af_array dB) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//creando copia de dB
	af_array b;
	af_copy_array(&b, dB);

	//2. Obteniendo parámetros necesarios

	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmL);
	size_t size_elmL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], colL);
	size_t size_colL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], rowL);
	size_t size_rowL = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, elmL);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_elmL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_elmL,
		NULL, &status);
	af_get_device_ptr((void**)d_elmL, elmL);

	cl_mem *d_colL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_colL,
		NULL, &status);
	af_get_device_ptr((void**)d_colL, colL);

	cl_mem *d_rowL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_rowL,
		NULL, &status);
	af_get_device_ptr((void**)d_rowL, rowL);

	cl_mem *d_b = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_colL,
		NULL, &status);
	af_get_device_ptr((void**)d_b, b);

	size_t program_length = strlen(ldlt_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context,
		1, (const char **)&ldlt_source, &program_length,
		&status);
	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "ldlt_sparse_c";
	else if (typef == f32)
		kernelName = "ldlt_sparse_c_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	int step = 0;
	int sstep = 0;
	int kk = 0;
	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_colL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_rowL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &kk);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_colL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_b);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &step);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &sstep);

	//6. ejecutando el kernel

	//resolviendo el sistema Ly=b (y=D*transpose(L)*x)
	sstep = 2;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
	for (int j = 0; j < size_colL - 1; j++) {

		clSetKernelArg(kernel, 10, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//resolviendo el sistema Dz=y, z=transpose(L)*x
	sstep++;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//resolviendo el sistema transpose(L)*x=z
	sstep++;
	clSetKernelArg(kernel, 11, sizeof(cl_int), &sstep);

	for (int j = 0; j < size_colL; j++) {

		clSetKernelArg(kernel, 10, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(elmL);
	af_unlock_array(colL);
	af_unlock_array(rowL);
	af_unlock_array(b);

	//copiando al argumento de salida
	af_copy_array(dC, b);

	af_release_array(b);
}

void AFire::fac_sparse_ldlt_sks(af_array elmA,
	af_array idxA) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//2. Obteniendo parámetros necesarios
	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmA);
	size_t size_elmL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], idxA);
	size_t size_idxL = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, elmA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_elmL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_elmL,
		NULL, &status);
	af_get_device_ptr((void**)d_elmL, elmA);

	cl_mem *d_idxL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_idxL,
		NULL, &status);
	af_get_device_ptr((void**)d_idxL, idxA);

	size_t program_length = strlen(ldlt_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context,
		1, (const char **)&ldlt_source, &program_length,
		&status);
	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "ldlt_sparse_sks";
	else if (typef == f32)
		kernelName = "ldlt_sparse_sks_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_idxL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_idxL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), 0);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);

	//6.Ejecutando el kernel
	for (int j = 0; j < size_idxL - 1; j++)
	{
		//Se modifican los elementos de la columna j a partir 
		//de la fila j + 1.
		//A(n,j)=A(n,j)/A(j,j) n>j
		int sstep = 0;
		clSetKernelArg(kernel, 6, sizeof(cl_int), &j);
		clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);

		//para una columna s, s > j, se modifican las filas n >= s
		//A(n,s)=A(n,s)-A(s,j)*A(n,j)*A(j,j) s>j, n>=s
		sstep++;
		clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(elmA);
	af_unlock_array(idxA);
}

void AFire::SELldlt_sparse_sks(af_array* dC, af_array elmL,
	af_array idxL, af_array dB) {
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//creando copia de dB
	af_array b;
	af_copy_array(&b, dB);

	//2. Obteniendo parámetros necesarios

	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], elmL);
	size_t size_elmL = _order[0];

	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], idxL);
	size_t size_idxL = _order[0];

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, elmL);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_elmL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, msize*size_elmL,
		NULL, &status);
	af_get_device_ptr((void**)d_elmL, elmL);

	cl_mem *d_idxL = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_ONLY, sizeof(int)*size_idxL,
		NULL, &status);
	af_get_device_ptr((void**)d_idxL, idxL);

	cl_mem d_b = (cl_mem)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_idxL,
		NULL, &status);
	af_get_device_ptr((void**)d_b, b);

	size_t program_length = strlen(ldlt_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context,
		1, (const char **)&ldlt_source, &program_length,
		&status);
	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "ldlt_sparse_sks";
	else if (typef == f32)
		kernelName = "ldlt_sparse_sks_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	cl_int kk = 0;
	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_elmL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_idxL);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &kk);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_idxL);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_b);
	clSetKernelArg(kernel, i++, msize*localWorkSize, 0);

	//6. ejecutando el kernel
	//resolviendo el sistema Ly=b (y=D*transpose(L)*x)
	cl_int sstep = 2;
	clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
	for (cl_int j = 0; j < size_idxL - 1; j++) {

		clSetKernelArg(kernel, 6, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//resolviendo el sistema Dz=y, z=transpose(L)*x
	sstep++;
	clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	//resolviendo el sistema transpose(L)*x=z
	sstep++;
	clSetKernelArg(kernel, 7, sizeof(cl_int), &sstep);
	for (cl_int j = 0; j < size_idxL; j++) {
		clSetKernelArg(kernel, 6, sizeof(cl_int), &j);
		clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
			&globalWorkSize, &localWorkSize, 0, NULL,
			NULL);
	}

	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(elmL);
	af_unlock_array(idxL);
	af_unlock_array(b);

	//copiando al argumento de salida
	af_copy_array(dC, b);

	af_release_array(b);
}
//---------------
//Pruebas y otros
//---------------
void AFire::fenceTest_sp(af::array &A, af::array &B) {
	//Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//obteniendo las referencias cl_mem de los objetos af::array
	cl_mem * d_A = A.device<cl_mem>();
	cl_mem * d_B = B.device<cl_mem>();
	//------------------------------------
	//detalles importantes sobre el kernel
	//------------------------------------
	/**/

	size_t order = (int)A.dims(0);

	size_t program_length = strlen(GJordan_source);
	int status = CL_SUCCESS;

	//creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&GJordan_source, &program_length, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);
	cl_kernel kernel = clCreateKernel(program, "fenceTest_sp", &status);

	// estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_B);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &order);

	size_t localWorkSize[] = { BLOCK_SIZE, BLOCK_SIZE };
	size_t globalWorkSize[] =
	{ shrRoundUp(localWorkSize[0], order),shrRoundUp(localWorkSize[1], order) };
	//ejecutando el Kernel
	clEnqueueNDRangeKernel(af_queue, kernel, 2, 0,
		globalWorkSize, localWorkSize, 0, NULL, NULL);

	//devolviendo el control de memoria af::array a ArrayFire 
	A.unlock();
	B.unlock();
}

void AFire::test_1(af::array &A, af::array &B) 
{
	//B = A;
	size_t len = A.dims(0);
	seq idx(0, len - 1, 1);
	af::copy(B, A, idx);
}

void AFire::test_2(af_array dA)
{
	//1. Obteniendo el dispositivo, contexto y la cola usada por ArrayFire
	//cl_context af_context;
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//2. Obteniendo parámetros necesarios

	//longitud de los vectores
	dim_t _order[AF_MAX_DIMS];
	af_get_dims(&_order[0], &_order[1], &_order[2],
		&_order[3], dA);
	size_t size_elmA = _order[0];

	af_array zeros;
	dim_t size[] = { size_elmA };
	af_constant(&zeros, 0, 1, size, s32);

	size_t localWorkSize = BLOCK_SIZE * BLOCK_SIZE;
	size_t globalWorkSize = localWorkSize * BLOCK_SIZE;

	int status = CL_SUCCESS;

	af_dtype typef;
	af_get_type(&typef, dA);

	int msize = 0;
	if (typef == f64)
		msize = sizeof(double);
	else if (typef == f32)
		msize = sizeof(float);
	else;

	//3.obteniendo las referencias cl_mem de los objetos af::array
	cl_mem *d_A = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, msize*size_elmA,
		NULL, &status);
	af_get_device_ptr((void**)d_A, dA);

	cl_mem *d_zeros = (cl_mem*)clCreateBuffer(af_context,
		CL_MEM_READ_WRITE, sizeof(int)*size_elmA,
		NULL, &status);
	af_get_device_ptr((void**)d_zeros, zeros);

	size_t program_length = strlen(gc_source);

	//4.creando el programa, construyendo el ejecutable y extrayendo el punto de entrada
	// para el Kernel
	cl_program program = clCreateProgramWithSource(af_context,
		1, (const char **)&gc_source, &program_length,
		&status);
	status = clBuildProgram(program, 1, &af_device_id,
		NULL, NULL, NULL);

	char* kernelName;
	if (typef == f64)
		kernelName = "prueba_1";
	else if (typef == f32)
		kernelName = "prueba_1_sp";
	else;
	cl_kernel kernel = clCreateKernel(program, kernelName,
		&status);

	// 5.estableciendo los argumentos
	int i = 0;
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_A);
	clSetKernelArg(kernel, i++, sizeof(cl_mem), d_zeros);
	clSetKernelArg(kernel, i++, sizeof(cl_int), &size_elmA);

	//6. ejecutando el kernel
	clEnqueueNDRangeKernel(af_queue, kernel, 1, 0,
		&globalWorkSize, &localWorkSize, 0, NULL,
		NULL);

	af_print_array(zeros);
	//7. devolviendo el control de memoria af::array a ArrayFire 
	af_unlock_array(dA);
	af_unlock_array(zeros);

	af_release_array(zeros);
}