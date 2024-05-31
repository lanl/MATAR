#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// ************ 1D ************ //

// Test size function
TEST(Test_1D_CArrayKokkos, size)
{
	const int size = 1000;
	CArrayKokkos<double> A(size);

	EXPECT_EQ(size, A.size());
}

// Test extent function
TEST(Test_1D_CArrayKokkos, extent)
{
	const int size = 1000;
	CArrayKokkos<double> A(size);

	int length = size;
	EXPECT_EQ(length, A.extent());
}

// Test dims function
TEST(Test_1D_CArrayKokkos, dims)
{
	const int size = 1000;
	CArrayKokkos<double> A(size);
	
	int dims = size;
	EXPECT_EQ(dims, A.dims(0));

	// Note: extend to other dims when initialized to zero
}

// Test order function
TEST(Test_1D_CArrayKokkos, order)
{
	const int size = 1000;
	CArrayKokkos<double> A(size);
	
	int order = 1;
	EXPECT_EQ(order, A.order());
}

// Test pointer function
TEST(Test_1D_CArrayKokkos, pointer)
{
	const int size = 1000;
	CArrayKokkos<double> A(size);

	auto a = A.get_kokkos_view();

	EXPECT_EQ(&a[0], A.pointer());
}


// ************ 2D ************ //

// Test size function
TEST(Test_2D_CArrayKokkos, size)
{
	const int size0 = 100;
	const int size1 = 50;
	CArrayKokkos<double> A(size0, size1);

	EXPECT_EQ(size0*size1, A.size());
}

// Test extent function
TEST(Test_2D_CArrayKokkos, extent)
{
	const int size0 = 100;
	const int size1 = 50;
	CArrayKokkos<double> A(size0, size1);

	int length = size0*size1;
	EXPECT_EQ(length, A.extent());
}

// Test dims function
TEST(Test_2D_CArrayKokkos, dims)
{
	const int size0 = 100;
	const int size1 = 50;
	CArrayKokkos<double> A(size0, size1);
	
	int dims0 = size0;
	int dims1 = size1;
	EXPECT_EQ(dims0, A.dims(0));
	EXPECT_EQ(dims1, A.dims(1));

	// Note: extend to other dims when initialized to zero
}

// Test order function
TEST(Test_2D_CArrayKokkos, order)
{
	const int size = 100;
	CArrayKokkos<double> A(size, size);
	
	int order = 2;
	EXPECT_EQ(order, A.order());
}

// Test pointer function
TEST(Test_2D_CArrayKokkos, pointer)
{
	const int size = 100;
	CArrayKokkos<double> A(size, size);

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
