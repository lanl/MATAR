#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace


// Test size function
TEST(Test_1D_CArrayKokkos, size)
{
	const int size = 1000;
	CArrayKokkos<double> A(size);

	FOR_ALL (i, 0, size, {
        A(i) = 1.0;
    }); // end parallel for

	EXPECT_EQ(size, A.size());
}

// Test extent function
TEST(Test_1D_CArrayKokkos, extent)
{
	const int size = 1000;
	CArrayKokkos<double> A(size);

	FOR_ALL (i, 0, size, {
        A(i) = 1.0;
    }); // end parallel for

	int length = size;
	EXPECT_EQ(length, A.extent());
}

// Test dims function
TEST(Test_1D_CArrayKokkos, dims)
{
	const int size = 1000;
	CArrayKokkos<double> A(size);

	FOR_ALL (i, 0, size, {
        A(i) = 1.0;
    }); // end parallel for
	
	int dims = size;
	EXPECT_EQ(dims, A.dims(0));

	// Note: extend to other dims when initialized to zero
}

// Test order function
TEST(Test_1D_CArrayKokkos, order)
{
	const int size = 1000;
	CArrayKokkos<double> A(size);

	FOR_ALL (i, 0, size, {
        A(i) = 1.0;
    }); // end parallel for
	
	int order = 1;
	EXPECT_EQ(order, A.order());
}

// Test pointer function
TEST(Test_1D_CArrayKokkos, pointer)
{
	const int size = 1000;
	CArrayKokkos<double> A(size);

	FOR_ALL (i, 0, size, {
        A(i) = 1.0;
    }); // end parallel for

	auto a = A.get_kokkos_view();

	EXPECT_EQ(&a[0], A.pointer());
}


int main(int argc, char* argv[])
{


	Kokkos::initialize(argc, argv);
    {  

		int result = 0;

		testing::InitGoogleTest(&argc, argv);

		result = RUN_ALL_TESTS();

		return result;

	}
	Kokkos::finalize();

}
