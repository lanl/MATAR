#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <memory> // for shared_ptr
#include <benchmark/benchmark.h>
#include "matar.h"

using namespace mtr; // matar namespace

// ------- vector vector multiply ------------- //
static void BM_CArrayDevice_1d_multiply(benchmark::State& state) 
{

    int size = state.range(0);

    CArrayDevice<double> A(size);
    CArrayDevice<double> B(size);
    CArrayDevice<double> C(size);

    FOR_ALL(i, 0, size, {
        A(i) = (double)i+1.0;
        B(i) = (double)i+2.0;
    });
    // Begin benchmarked section
    for (auto _ : state){
        FOR_ALL(i, 0, size, {
            C(i) = A(i)*B(i);
        });
    } // end benchmarked section

    // Kokkos::finalize();

}
BENCHMARK(BM_CArrayDevice_1d_multiply)
->Unit(benchmark::kMillisecond)
->Name("Benchmark Multiplying 2 1D CArrayDevice of size ")
->RangeMultiplier(2)->Range(1<<12, 1<<20);

// ------- vector vector dot product ------------- //
static void BM_CArrayDevice_vec_vec_dot(benchmark::State& state) 
{
    int size = state.range(0);

    CArrayDevice<double> A(size);
    CArrayDevice<double> B(size);
    double C = 0.0;

    FOR_ALL(i, 0, size, {
        A(i) = (double)i+1.0;
        B(i) = (double)i+2.0;
    });


    // Begin benchmarked section
    for (auto _ : state){

        double loc_sum = 0;
        double C  = 0;
        REDUCE_SUM(i, 0, size,
            loc_sum, {
            loc_sum += A(i)*B(i);
        }, C);
    } // end benchmarked section
}
BENCHMARK(BM_CArrayDevice_vec_vec_dot)
->Unit(benchmark::kMillisecond)
->Name("Benchmark dot product of 2 1D CArrayDevice of size ")
->RangeMultiplier(2)->Range(1<<12, 1<<20);


// ------- matrix matrix multiply ------------- //
static void BM_CArrayDevice_mat_mat_multiply(benchmark::State& state) 
{
    int size = state.range(0);

    CArrayDevice<double> A(size, size);
    CArrayDevice<double> B(size, size);
    CArrayDevice<double> C(size, size);

    FOR_ALL(i, 0, size,
            j, 0, size, {
        A(i,j) = (double)i+(double)j+1.0;
        B(i,j) = (double)i+(double)j+2.0;
        C(i,j) = 0.0;
    });

    // Begin benchmarked section
    for (auto _ : state){
        
        FOR_ALL(i, 0, size,
                j, 0, size, {
            for(int k = 0; k < size; k++){
                C(i,k) += A(i,j)*B(j,k);
            }
        });
    } // end benchmarked section
}
BENCHMARK(BM_CArrayDevice_mat_mat_multiply)
->Unit(benchmark::kMillisecond)
->Name("Benchmark matrix-matrix multiply of CArrayDevice of size ")
->RangeMultiplier(2)->Range(1<<3, 1<<10);


// Run Benchmarks
int main(int argc, char** argv)
{
   Kokkos::initialize();
   ::benchmark::Initialize(&argc, argv);
   ::benchmark::RunSpecifiedBenchmarks();
   Kokkos::finalize();
}
