#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <memory> // for shared_ptr
#include <benchmark/benchmark.h>
#include "matar.h"

using namespace mtr; // matar namespace

// ------- vector vector multiply ------------- //
static void BM_CArray_1d_multiply(benchmark::State& state) 
{
    // const int size = 4000;

    int size = state.range(0);

    CArray<double> A(size);
    CArray<double> B(size);
    CArray<double> C(size);

    for(int i=0; i<size; i++){
        A(i) = (double)i+1.0;
        B(i) = (double)i+2.0;
    }
    
    // Begin benchmarked section
    for (auto _ : state){
        for(int i=0; i<size; i++){
            C(i) = A(i)*B(i);
        }
    } // end benchmarked section
}
BENCHMARK(BM_CArray_1d_multiply)
->Unit(benchmark::kMillisecond)
->Name("Benchmark Multiplying 2 1D CArrays of size ")
->RangeMultiplier(2)->Range(1<<12, 1<<20);


// ------- vector vector dot product ------------- //
static void BM_Carray_vec_vec_dot(benchmark::State& state) 
{
    int size = state.range(0);

    // Begin benchmarked section
    for (auto _ : state){
        CArray<double> A(size);
        CArray<double> B(size);
        double C = 0.0;

        for(int i = 0; i < size; i++){
            A(i) = (double)i+1.0;
            B(i) = (double)i+2.0;
        }

        for(int i = 0; i < size; i++){
            C += A(i)*B(i);
        }
    } // end benchmarked section
}
BENCHMARK(BM_CArray_1d_multiply)
->Unit(benchmark::kMillisecond)
->Name("Benchmark dot product of 2 1D CArrays of size ")
->RangeMultiplier(2)->Range(1<<12, 1<<20);

// ------- matrix matrix multiply ------------- //
static void BM_CArray_mat_mat_multiply(benchmark::State& state) 
{
    int size = state.range(0);

    // Begin benchmarked section
    for (auto _ : state){
        CArray<double> A(size, size);
        CArray<double> B(size, size);
        CArray<double> C(size, size);

        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++){
                A(i,j) = (double)i+(double)j+1.0;
                B(i,j) = (double)i+(double)j+2.0;
                C(i,j) = 0.0;
            }
        }

        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++){
                for(int k = 0; k < size; k++){
                    C(i,k) += A(i,j)*B(j,k);
                }
            }
        }
    } // end benchmarked section
}
BENCHMARK(BM_CArray_mat_mat_multiply)
->Unit(benchmark::kMillisecond)
->Name("Benchmark matrix-matrix multiply of CArrays of size ")
->RangeMultiplier(2)->Range(1<<3, 1<<10);

// Run benchmarks
int main(int argc, char** argv)
{
   Kokkos::initialize();
   ::benchmark::Initialize(&argc, argv);
   ::benchmark::RunSpecifiedBenchmarks();
   Kokkos::finalize();
}
