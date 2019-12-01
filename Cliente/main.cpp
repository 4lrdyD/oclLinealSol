#include "Header.h"
#include "Common.h"

using namespace af;
void testloop(int orden, int smax, af_dtype typef);
int main2()
{
	af_dtype typef = f64;
	int orden = 5;
	int smax = 7;
	int nite = 1;

	int idat = 0;
	std::cout <<
		"Ingresar datos?: 1=si, 2=solo precisión, 3=loop" 
		<<std::endl;
	std::cin >> idat;

	if (idat > 0)
	{
		std::cout << "1=float, 2=double" << std::endl;
		int ntype;
		std::cin >> ntype;
		typef = (ntype == 1 ? f32 : f64);
		std::cout << "Eligió: " <<
			(typef == f32 ? "float" : "double") << std::endl;

		idat += 1;
		if (idat == 2)
		{
			std::cout << "orden:" << std::endl;
			std::cin >> orden;

			std::cout << "ancho máximo que se pueda imprimir:" << std::endl;
			std::cin >> smax;
		}
		else if (idat == 4)
		{
			std::cout << "tamaño de bloque:" <<
				BLOCK_SIZE<<std::endl;

			std::cout << "orden inicial:" << std::endl;
			std::cin >> orden;

			std::cout << "numero de iteraciones:" << std::endl;
			std::cin >> nite;

			std::cout << "ancho máximo que se pueda imprimir:" << std::endl;
			std::cin >> smax;
		}
		else;
	}

	if (idat == 4)
	{
		for (int i = orden; i < orden + nite; i++)
		{
			std::cout << "             ORDEN: " << i << std::endl;
			std::cout << "             ------------" << std::endl;
			testloop(i, smax, typef);
		}
		orden += nite;
		int cont = 1;
		while (cont == 1)
		{
			std::cout << "siguiente?: 1=si" << std::endl;
			std::cin >> cont;
			if(cont==1)
				testloop(orden, smax, typef);
			orden += 1;
		}
	}
	else 
	{
		testloop(orden, smax, typef);
	}
	return 0;
}

void testloop(int orden, int smax, af_dtype typef)
{
	/*
seq s1(0, 4, 1);//rango que servirá para extraer submatrices
seq s2(0, 4, 1);
array dD = randn(siA1, siB2);
AFire::mulaf(dA, dB, dD, siA2, siB2, siA1);
af_print(dD(s1, s2));
std::cout << dA.dims(0)<<" "<<dA.dims(1) << std::endl;
*/
	//multiplicación
	std::cout << "--------------------------" << std::endl;
	std::cout << "multiplicación de matrices" << std::endl;
	std::cout << "--------------------------" << std::endl;
	array dA = randn(orden, orden, typef);
	array dB = randn(orden, orden, typef);
	array dC = matmul(dA, dB);//AF
	if (dA.dims(1) < smax)
		af_print(dA);
	if (dB.dims(1) < smax)
		af_print(dB);
	std::cout << "ArrayFire:matmul" << std::endl;
	if (dC.dims(1) < smax)
		af_print(dC);
	std::cout << "Propio:mulaf" << std::endl;
	af_array ddD;
	AFire::mulaf(&ddD, dA.get(), dB.get());
	if (dB.dims(1) < smax)
		af_print_array(ddD);

	/*farray dD = randn(orden, orden, typef);
	typef == f64 ? AFire::mulaf(dD, dA, dB)//propio
		: AFire::mulaf_sp(dD, dA, dB);

	if (dD.dims(1) < smax)
		af_print(dD);*/

	//resolviendo SEL's
	std::cout << "-----------------" << std::endl;
	std::cout << "Resolviendo SEL's" << std::endl;
	std::cout << "-----------------" << std::endl;
	array dD;
	dD = randn(orden, orden, typef);
	array dF = sum(dD, 1);

	if (dD.dims(1) < smax)
		af_print(dD);
	if (dF.dims(1) < smax)
		af_print(dF);

	std::cout << "ArrayFire" << std::endl;
	//solve de ArrayFire
	array dG = solve(dD, dF);
	if (dG.dims(1) < smax)
		af_print(dG);

	//solve propio
	std::cout << "SELgj_f" << std::endl;
	af_array dHH;
	AFire::SELgj_f(&dHH, dD.get(), dF.get());
	af_print_array(dHH);

	std::cout << "SELgj_c" << std::endl;
	af_array dI;
	AFire::SELgj_c(&dI, dD.get(), dF.get());
	af_print_array(dI);

	std::cout << "SELgj_f2d" << std::endl;
	af_array dK;
	AFire::SELgj_f2d(&dK, dD.get(), dF.get());
	af_print_array(dK);

	std::cout << "SELgj_c2d" << std::endl;
	af_array dJ;
	AFire::SELgj_c2d(&dJ, dD.get(), dF.get());
	af_print_array(dJ);

	/*std::cout << "SELgj_fshr" << std::endl;
	array dM = af::randu(orden, typef);
	typef == f64 ? AFire::SELgj_fshr(dM, dD, dF)
		: AFire::SELgj_fshr_sp(dM, dD, dF);
	if (dM.dims(1) < smax)
		af_print(dM);
	
	/*std::cout << "prueba" << std::endl;
	array dL = af::randu(orden, typef);
	seq seq1(10, orden + 9, 1);
	dF = array(seq1, orden, 1);
	if (dF.dims(1) < smax)
		af_print(dF);
	if (dL.dims(1) < smax)
		af_print(dL);
	typef == f64 ? AFire::prueba_shr(dL, dD, dF)
		: AFire::prueba_shr_sp(dL, dD, dF);
	if (dF.dims(1) < smax)
		af_print(dF);
	if (dL.dims(1) < smax)
		af_print(dL);*/
}

int main_sparse_c()//sparse_c
{
	/*float _M[] = { 5,1,-2,0,1,2,0,0,-2,0,4,1,0,0,1,3 };
	float _b[] = { 1,5,14,15};
	array M(4, 4, _M);
	array b(4, 1, _b);
	af_print(M);
	af_print(b);*/

	float _M[] = { 3, 1, 5, 1, 0, 3, 0,
		1, 1, 0, 5, 0, 0, 0,
		5, 0, 2, 0, 4, 0, 0,
		1, 5, 0, 9, 0, 0, 6,
		0, 0, 4, 0, 2, 0, 11,
		3, 0, 0, 0, 0, 5, 0,
		0, 0, 0, 6, 11, 0, 1 };

	float _b[] = { 9.0000,
			5.0000,
			14.0000,
			15.0000,
			12,
			10,
			13 };

		array M(7, 7, _M);
		array b(7, 1, _b);
		af_print(M);
		af_print(b);

	/*float _elmA[] = { 5,1,-2,2,4,1,3 };
	int _colA[] = { 0,3,4,6 };
	int _rowA[] = { 1,2,3};
	float _elmL[] = { 0,0,0,0,0,0,0,0};
	int _colL[] = { 0,3,5,7 };
	int _rowL[] = { 1,  2,2, 3 };

	array elmA(7, 1, _elmA);
	array colA(4, 1, _colA);
	array rowA(3, 1, _rowA);
	array elmL(8, 1, _elmL);
	array colL(4, 1, _colL);
	array rowL(4, 1, _rowL);*/

	float _elmA[] = { 3,1,5,1,3,1,5,2,4,9,6,2,11,5,1 };
	int _colA[] = { 0,5,7,9,11,13,14 };
	int _rowA[] = { 1,2,3,5,3,4,6,6 };
	float _elmL[] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0 };
	int _colL[] = { 0,5,9,13,17,20,22 };
	int _rowL[] = { 1,  2,  3,  5,  2,  3,
		5,  3, 4,  5,  4,  5,  6,  5,  6,  6 };

	array elmA(15, 1, _elmA);
	array colA(7, 1, _colA);
	array rowA(8, 1, _rowA);
	array elmL(23, 1, _elmL);
	array colL(7, 1, _colL);
	array rowL(16, 1, _rowL);

	af_array mul;
	af_matmul(&mul, M.get(),b.get(),AF_MAT_NONE,
		AF_MAT_NONE);
	af_print_array(mul);

	af_release_array(mul);
	AFire::sparse_mat_vec_mul(&mul, elmA.get(),
		colA.get(), rowA.get(), b.get());
	af_print_array(mul);

	return 0;
}

int main_sparse_sks()//sparse_sks
{
	/*double _M[] = { 5,1,-2,0,1,2,0,0,-2,0,4,1,0,0,1,3 };
	double _b[] = { 1,5,14,15 };
	array M(4, 4, _M);
	array b(4, 1, _b);
	af_print(M);
	af_print(b);*/

	double _M[] = { 3, 1, 5, 1, 0, 3, 0,
		1, 1, 0, 5, 0, 0, 0,
		5, 0, 2, 0, 4, 0, 0,
		1, 5, 0, 9, 0, 0, 6,
		0, 0, 4, 0, 2, 0, 11,
		3, 0, 0, 0, 0, 5, 0,
		0, 0, 0, 6, 11, 0, 1 };

	double _b[] = { 9.0000,
			5.0000,
			14.0000,
			15.0000,
			12,
			10,
			13 };

		array M(7, 7, _M);
		array b(7, 1, _b);
		af_print(M);
		af_print(b);

	/*double _elmA[] = { 5,1,-2,2,0,4,1,3 };
	int _idxA[] = { 0,3,5,7 };

	array elmA(8, 1, _elmA);
	array idxA(4, 1, _idxA);*/

	double _elmA[] = { 3.0,
  1.0,
  5.0,
  1.0,
  0.0,
  3.0,
  1.0,
  0.0,
  5.0,
  0.0,
  0.0,
  2.0,
  0.0,
  4.0,
  0.0,
  9.0,
  0.0,
  0.0,
  6.0,
  2.0,
  0.0,
 11.0,
  5.0,
  0.0,
  1.0 };
	int _idxA[] = { 0,
  6,
 11,
 15,
 19,
 22,
 24};

	array elmA(25, 1, _elmA);
	array idxA(7, 1, _idxA);

	af_array mul;
	AFire::SELgc_sparse_sks(&mul, elmA.get(),
		idxA.get(), b.get(),1e-6);
	af_print_array(mul);
	
	af_release_array(mul);
	AFire::SELgj_c(&mul, M.get(), b.get());
	af_print_array(mul);

	af_release_array(mul);
	AFire::global_sync_test(&mul, M.get(), b.get());
	af_print_array(mul);

	return 0;
}

int main1()
{
	double _M[] = { 4.,	4.,	4.,	2.,	6.,	7.,	2.,
5.,	2.,	4.,	6.,	2.,	3.,	4.,
5.,	6.,	2.,	7.,	8.,	2.,	6.,
10.,	3.,	1.,	7.,	9.,	1.,	4.,
10.,	5.,	5.,	3.,	2.,	9.,	3.,
4.,	9.,	7.,	8.,	5.,	2.,	7.,
7.,	8.,	5.,	6.,	2.,	8.,	2.
	};

	double _b[] = { 118.,
101.,
145.,
126.,
132.,
161.,
134.
	};

	array __M(7, 7, _M);
	array M = transpose(__M);
	array b(7, 1, _b);
	af_print(M);
	af_print(b);

	af_array mul;
	AFire::SELgj_c(&mul, M.get(), b.get());
	af_print_array(mul);

	af_release_array(mul);
	AFire::global_sync_test(&mul, M.get(), b.get());
	af_print_array(mul);

	return 0;
}

void sum(float *gC, float *gA, float *gB, int orden) 
{
	for (int id = 0; id < orden; id++) {
		gC[id] = gA[id] + gB[id];
	}
}
int main()
{
	int row_[] = { 0, 1, 4, 0, 1, 2, 3, 4, 1, 2,
		5, 1, 3, 5, 0, 1, 4, 5, 6, 2, 3, 4, 5, 6,
		4, 5, 6 };
	int col_[] = { 0, 0, 0, 1,  1,  1, 1, 1, 2, 2,
		2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
		6, 6, 6 };
	double nz_[] = { 1, 2, 3, 2, 2, 5, 6, 7, 5, 4,
	4, 6, 3, 6, 3, 7, 7, 2, 9, 4, 6, 2, 9, 6,
	9, 6, 9 };

	array row(27, 1, row_);
	array col(27, 1, col_);
	array nz(27, 1, nz_);
	
	af_array index;
	AFire::csc_util_1(&index, row.get(),
		col.get());
	af_print_array(index);

	af_array del;
	AFire::csc_util_2(&del, index, row.get());
	af_print_array(del);

	int del2_[] = { 0, 1, 2, 3, 5, 8, 10 };
	array del2(7, 1, del2_);

	AFire::csc_util_3(del2.get(), index);
	af_print_array(index);

	af_array csc;
	dim_t d_order[] = { 17 };
	af_constant(&csc, 0, 1, d_order, f64);

	af_array rowc;
	dim_t d_order_[] = { 10 };
	af_constant(&rowc, 0, 1, d_order_, s32);

	AFire::csc_util_4(csc, rowc, index, nz.get(),
		row.get(), del2.get());
	af_print_array(csc);
	af_print_array(rowc);
	/*double real;
	double imag;

	af_sum_all(&real, &imag, index);
	std::cout << real << std::endl;

	af_array sks;
	dim_t d_order[] = { (int)real };
	af_constant(&sks, 0, 1, d_order, f32);

	int index_[] = { 0, 5, 9, 13, 16, 19, 21};

	array idx(7, 1, index_);

	AFire::sks_util_2(sks,idx.get(),nz.get(),
		row.get(), col.get());
	af_print_array(sks);
	*/
	
}