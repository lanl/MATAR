#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr;  // matar namespace

namespace {
// A KOKKOS_LAMBDA cannot appear inside TEST()'s private TestBody() -- nvcc
// rejects extended __device__ lambdas in a function with private class access
// -- so the RUN block that fills a view lives in this free function instead.
// The view is captured by value into the kernel, so set_values must be const.
template <typename ViewType, typename T>
inline void helper_set_values(const ViewType& view, T val) {
    RUN({
        view.set_values(val);
    });
    MATAR_FENCE();
}
}  // namespace

// Helper function to create matrices of different dimensions
ViewFMatrixKokkos<double> return_ViewFMatrixKokkos(int dims, std::vector<int> sizes, double* data) {
    switch (dims) {
        case 1:
            return ViewFMatrixKokkos<double>(data, sizes[0]);
        case 2:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1]);
        case 3:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2]);
        case 4:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3]);
        case 5:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
        case 6:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5]);
        case 7:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6]);
        default:
            return ViewFMatrixKokkos<double>();
    }
}

// Test default constructor
TEST(Test_ViewFMatrixKokkos, default_constructor) {
    ViewFMatrixKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.order(), 0);
    EXPECT_EQ(A.pointer(), nullptr);
}

// Test size function
TEST(Test_ViewFMatrixKokkos, size) {
    const int size = 100;
    double* data   = new double[size * size];
    ViewFMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(size * size, A.size());
    delete[] data;
}

// Test extent function
TEST(Test_ViewFMatrixKokkos, extent) {
    const int size = 100;
    double* data   = new double[size * size];
    ViewFMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(size * size, A.extent());
    delete[] data;
}

// Test dims function
TEST(Test_ViewFMatrixKokkos, dims) {
    const int size = 100;
    double* data   = new double[size * size * size];
    ViewFMatrixKokkos<double> A(data, size, size, size);
    EXPECT_EQ(size, A.dims(1));
    EXPECT_EQ(size, A.dims(2));
    EXPECT_EQ(size, A.dims(3));
    delete[] data;
}

// Test order function
TEST(Test_ViewFMatrixKokkos, order) {
    const int size = 100;
    double* data   = new double[size * size * size];
    ViewFMatrixKokkos<double> A(data, size, size, size);
    EXPECT_EQ(3, A.order());
    delete[] data;
}

// Test pointer function
TEST(Test_ViewFMatrixKokkos, pointer) {
    const int size = 100;
    double* data   = new double[size * size];
    ViewFMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(data, A.pointer());
    delete[] data;
}

// Test set_values function
TEST(Test_ViewFMatrixKokkos, set_values) {
    const int size = 100;
    Kokkos::View<double*> dev_data("dev_data", size * size);
    ViewFMatrixKokkos<double> A(dev_data.data(), size, size);

    helper_set_values(A, 42.0);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
    for (int i = 0; i < size * size; i++) {
        EXPECT_EQ(42.0, h(i));
    }
}

// Test operator access
TEST(Test_ViewFMatrixKokkos, operator_access) {
    const int size = 10;
    double* data   = new double[size * size * size];
    ViewFMatrixKokkos<double> A(data, size, size, size);

    // Test 3D access
    for (int i = 1; i <= size; i++) {
        for (int j = 1; j <= size; j++) {
            for (int k = 1; k <= size; k++) {
                A(i, j, k) = i * 100 + j * 10 + k;
            }
        }
    }

    for (int i = 1; i <= size; i++) {
        for (int j = 1; j <= size; j++) {
            for (int k = 1; k <= size; k++) {
                EXPECT_EQ(i * 100 + j * 10 + k, A(i, j, k));
            }
        }
    }
    delete[] data;
}

#ifndef NDEBUG

// Test bounds checking
TEST(Test_ViewFMatrixKokkos, bounds_checking) {
    const int size = 10;
    double* data   = new double[size * size];
    ViewFMatrixKokkos<double> A(data, size, size);

    // Test valid access
    A(5, 5) = 42.0;
    EXPECT_EQ(42.0, A(5, 5));

    // Test invalid access - should throw
    EXPECT_DEATH(A(0, 0), ".*");
    delete[] data;
}
#endif

// Test different types
TEST(Test_ViewFMatrixKokkos, different_types) {
    const int size = 10;

    // Test int
    {
        Kokkos::View<int*> dev_data("int_data", size * size);
        ViewFMatrixKokkos<int> A(dev_data.data(), size, size);
        helper_set_values(A, 42);
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_EQ(42, h(0));
    }

    // Test float
    {
        Kokkos::View<float*> dev_data("float_data", size * size);
        ViewFMatrixKokkos<float> B(dev_data.data(), size, size);
        helper_set_values(B, 42.0f);
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_EQ(42.0f, h(0));
    }

    // Test bool
    {
        Kokkos::View<bool*> dev_data("bool_data", size * size);
        ViewFMatrixKokkos<bool> C(dev_data.data(), size, size);
        helper_set_values(C, true);
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_EQ(true, h(0));
    }
}

// Test RAII behavior: a ViewFMatrixKokkos is a non-owning alias of memory that
// a MATAR container owns, so constructing and destroying the view must never
// change the owner's reference count or the contents of the allocation.
TEST(Test_ViewFMatrixKokkos, raii) {
    const int size = 100;

    FMatrixKokkos<double> owner(size, size, "raii_owner");
    owner.set_values(1.0);
    MATAR_FENCE();

    // Hold a second reference to the owner's allocation so its reference count
    // is observable: owner + owner_ref == 2.
    auto owner_ref = owner.get_kokkos_view();
    EXPECT_EQ(2, owner_ref.use_count());

    double* const backing = owner.pointer();
    {
        ViewFMatrixKokkos<double> A(backing, size, size);

        // The view aliases the owner's memory rather than allocating its own.
        EXPECT_EQ(backing, A.pointer());
        EXPECT_EQ(owner.size(), A.size());
        EXPECT_EQ(2, owner_ref.use_count());

        helper_set_values(A, 42.0);
    }  // A is destroyed here

    // Destroying the view released nothing: the allocation is still owned only
    // by owner and owner_ref.
    EXPECT_EQ(2, owner_ref.use_count());

    // The data A wrote survived A's destruction, untouched.
    auto host_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, owner_ref);
    for (int i = 0; i < size * size; i++) {
        EXPECT_EQ(42.0, host_ref(i));
    }

    // A fresh view over the same memory reads back what the destroyed view wrote.
    Kokkos::View<double*> probe("probe", 1);
    ViewFMatrixKokkos<double> B(backing, size, size);
    RUN({
        probe(0) = B(1, 1);
    });
    MATAR_FENCE();
    auto host_probe = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, probe);
    EXPECT_EQ(42.0, host_probe(0));

    // The owning container is what implements RAII: releasing its reference is
    // tied to its scope, while any view of it holds no reference at all.
    decltype(owner.get_kokkos_view()) scoped_ref;
    {
        FMatrixKokkos<double> scoped(size, size, "raii_scoped");
        scoped_ref = scoped.get_kokkos_view();
        EXPECT_EQ(2, scoped_ref.use_count());

        ViewFMatrixKokkos<double> C(scoped.pointer(), size, size);
        helper_set_values(C, 7.0);
        EXPECT_EQ(2, scoped_ref.use_count());
    }  // scoped is destroyed here

    // Only our own reference is left, so scoped's destructor gave up its
    // reference and C never held one.
    EXPECT_EQ(1, scoped_ref.use_count());
}
