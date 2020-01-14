#include "Header.h"
#include "Common.h"

using namespace af;
int main() {
	af_array a;
	af_array b;
	dim_t size1[] = { 12,12 };
	dim_t size2[] = { 12};
	af_randu(&a, 2, size1, f64);
	af_constant(&b, 1, 1, size2, f64);
	af_print_array(b);

	af_array out;
	AFire::test_2(&out, a, b);
	af_print_array(out);

}