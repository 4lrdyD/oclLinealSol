#include "Header.h"
#include "Common.h"

using namespace af;
int main() {
	
	float _elmA[] = { 2,2,0,0,3,4,5,6,7,15,
		0,0,4,200,0,6,
	200,2,9,100,6,18 };
	int _idxA[] = { 0,5,9,13,16,19,21 };
	float B[] = { 7.0,
		24.0,
		24.0,
		212.0,
		221.0,
		118.0,
		33.0 };

	af_array elmA;
	af_array idxA;
	af_array b;
	dim_t size1[] = {22};
	dim_t size2[] = {7};

	af_create_array(&elmA, _elmA, 1, size1, f32);
	af_print_array(elmA);
	af_create_array(&idxA, _idxA, 1, size2, s32);
	af_print_array(idxA);
	af_create_array(&b, B, 1, size2, f32);
	af_print_array(b);

	AFire::SELgc_sparse_sks(elmA, idxA, b, 1e-6);
	af_print_array(b);

}