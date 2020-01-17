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

	AFire::SEL_gc(a, b, 1e-6);
	af_print_array(b);

	double n1 = 5.456467;
	float n2 = -1.456456;
	std::cout << (n1 + n2) << std::endl;
}