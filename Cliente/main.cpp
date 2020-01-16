#include "Header.h"
#include "Common.h"

using namespace af;
int main() {
	double A[]={2.0,2.0,0.0,0.0,3.0,0.0,0.0,
		2.0,  4.0,   5.0,    6.0, 7.0, 0.0, 0.0,
		0.0,  5.0,  15.0,    0.0, 0.0, 4.0, 0.0,
		0.0,  6.0,   0.0,  200.0,    0.0,6.0,0.0,
		3.0,  7.0,   0.0,    0.0,  200.0,2.0, 9.0,
		0.0,  0.0,   4.0,    6.0,    2.0,100.0,6.0,
		0.0,  0.0,   0.0,    0.0,    9.0, 6.0,  18.0};
	double B[] = { 7.0,
		24.0,
		24.0,
		212.0,
		221.0,
		118.0,
		33.0 };

	af_array a;
	af_array b;
	dim_t size1[] = { 7,7 };
	dim_t size2[] = { 7};

	af_create_array(&a, A, 2, size1, f64);
	af_print_array(a);
	af_create_array(&b, B, 1, size2, f64);
	af_print_array(b);

	af_array n1;
	af_matmul(&n1, a, b, AF_MAT_NONE, AF_MAT_NONE);
	af_array r;
	af_sub(&r, b, n1, false);
	
	af_array z;
	af_matmul(&z, a, r, AF_MAT_NONE, AF_MAT_NONE);

	af_array p;
	af_array a_;
	af_array rp;
	af_array pz;
	af_array ap;
	af_array x;
	af_copy_array(&p, r);
	af_matmul(&rp, r, p, AF_MAT_TRANS, AF_MAT_NONE);
	af_matmul(&pz, p, z, AF_MAT_TRANS, AF_MAT_NONE);
	af_div(&a_, rp, pz, false);

	af_mul(&ap, a_, p, true);
	af_add(&x, b, ap, false);

	af_array copyr;
	af_array axz;
	af_array rtxz;
	af_array ptxz;
	af_array rsp;
	af_array B_;
	af_array zero;
	dim_t d_order[] = { 1 };
	af_constant(&zero, 0, 1, d_order, f64);

	af_array Bxp;
	af_array rtxp;
	af_array x0;
	af_array axp;
	//for (int j = 0; j < 7; j++) {
		af_copy_array(&copyr, r);
		af_mul(&axz, a_, z, true);
		af_sub(&r, copyr, axz, false);
		af_print_array(r);
		double _norm;
		af_norm(&_norm, r, AF_NORM_EUCLID, 1, 1);
		std::cout << _norm << std::endl;

		af_matmul(&rtxz, r, z, AF_MAT_TRANS, AF_MAT_NONE);
		af_matmul(&ptxz, p, z, AF_MAT_TRANS, AF_MAT_NONE);
		af_div(&rsp, rtxz, ptxz, false);
		af_sub(&B_, zero, rsp, false);

		af_mul(&Bxp, B_, p, true);
		af_add(&p, r, Bxp, false);
		af_print_array(p);

		af_matmul(&z, a, p, AF_MAT_NONE, AF_MAT_NONE);
		af_print_array(z);

		af_matmul(&rtxp, r, p, AF_MAT_TRANS, AF_MAT_NONE);
		af_matmul(&ptxz, p, z, AF_MAT_TRANS, AF_MAT_NONE);
		af_div(&a_, rtxp, ptxz, false);
		af_print_array(a_);

		af_mul(&axp, a_, p, true);
		af_copy_array(&x0, x);
		af_add(&x, x0, axp, false);
		af_print_array(x);

	//}


	std::cout << "===================" << std::endl;
	af_print_array(b);
	af_array out;
	AFire::test_2(&out, a, b);
	af_print_array(b);

}