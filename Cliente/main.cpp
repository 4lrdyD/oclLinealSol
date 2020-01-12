#include "Header.h"
#include "Common.h"

using namespace af;
int main() {
	af_array a;
	af_array b;
	dim_t size1[] = { 1200,1200 };
	dim_t size2[] = { 1200};
	af_randu(&a, 2, size1, f32);
	af_randu(&b, 1, size2, f32);

	af_array A;
	af_matmul(&A, a, a, AF_MAT_TRANS, AF_MAT_NONE);

	af_array out1;
	af_matmul(&out1, A, b, AF_MAT_NONE,
		AF_MAT_NONE);

	af::timer::start();
	af_array out2;
	AFire::test_2(&out2, A, b);
	double k=af::timer::stop();
	std::cout << k << std::endl;

	af_array sub;
	af_sub(&sub, out1, out2, false);
	double norm;
	af_norm(&norm, sub, AF_NORM_EUCLID, 1, 1);
	std::cout << norm << std::endl;
	
}