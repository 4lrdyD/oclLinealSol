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
	af_print_array(r);
	
	af_array z;
	af_matmul(&z, a, r, AF_MAT_NONE, AF_MAT_NONE);
	af_print_array(z);

	af_array p;
	af_array a_;
	af_array rp;
	af_array pz;
	af_array ap;
	af_array t;
	af_copy_array(&p, r);
	af_matmul(&rp, r, p, AF_MAT_TRANS, AF_MAT_NONE);
	af_matmul(&pz, p, z, AF_MAT_TRANS, AF_MAT_NONE);
	af_div(&a_, rp, pz, false);
	af_print_array(a_);

	af_mul(&ap, a_, p, true);
	af_add(&t, b, ap, false);
	af_print_array(t);
	std::cout << 1e-6 << std::endl;

	af_array sol;
	AFire::SELchol_c(&sol, a, b);
	af_print_array(sol);


	af_array out;
	AFire::test_2(&out, a, b);

}