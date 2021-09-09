/**********************************************************************************************
 This file has suite of MACROS to build serial and parallel loops that are more readable and
 are written with the same syntax. The parallel loops use kokkos (i.e., the MACROS hide the
 complexity) and the serial loops are done using functions located in this file. The goal is to
 help users add kokkos to their code projects for performance portability across architectures.

 The loop order with the MACRO enforces the inner loop varies the fastest and the outer most
 loop varies the slowest.  Optiminal performance will be achieved by ensureing the loop indices
 align with the access pattern of the MATAR datatype.
 
 1.  The syntax to use the FOR_ALL MACRO is as follows:

 // parallelization over a single loop
 FOR_ALL(k, 0, 10,
        { loop contents is here });

 // parallellization over two loops
 FOR_ALL(m, 0, 3,
         n, 0, 3,
        { loop contents is here });

 // parallellization over two loops
 FOR_ALL(i, 0, 3,
         j, 0, 3,
         k, 0, 3,
        { loop contents is here });

 2.  The syntax to use the FOR_REDUCE is as follows:

 // reduce over a single loop
 FOR_REDUCE(i, 0, 100,
            local_answer,
           { loop contents is here }, answer);

 FOR_REDUCE(i, 0, 100,
            j, 0, 100,
            local_answer,
           { loop contents is here }, answer);
 
 FOR_REDUCE(i, 0, 100,
            j, 0, 100,
            k, 0, 100,
            local_answer,
           { loop contents is here }, answer);
 
 **********************************************************************************************/


#include <stdio.h>
#include <iostream>
#include <matar.h>


// -----------------------------------------
// The for_all and for_reduce functions that
// are used with the non-kokkos MACROS
// -----------------------------------------


template <typename F>
void for_all (int i_start, int i_end,
              const F &lambda_fcn){
    
        for (int i=i_start; i<i_end; i++){
            lambda_fcn(i);
        }
    
}; // end for_all


template <typename F>
void for_all (int i_start, int i_end,
              int j_start, int j_end,
              const F &lambda_fcn){
    
    for (int i=i_start; i<i_end; i++){
        for (int j=j_start; j<j_end; j++){
            lambda_fcn(i,j);
        }
    }
    
}; // end for_all


template <typename F>
void for_all (int i_start, int i_end,
              int j_start, int j_end,
              int k_start, int k_end,
              const F &lambda_fcn){
    
    for (int i=i_start; i<i_end; i++){
        for (int j=j_start; j<j_end; j++){
            for (int k=k_start; k<k_end; k++){
                lambda_fcn(i,j,k);
            }
        }
    }
    
}; // end for_all


template <typename T, typename F>
void for_reduce (int i_start, int i_end, T &var,
                 const F &lambda_fcn, T &result){
    
    for (int i=i_start; i<i_end; i++){
        lambda_fcn(i, var);
    }
    result = var;
};  // end for_reduce


template <typename T, typename F>
void for_reduce (int i_start, int i_end,
                 int j_start, int j_end,
                 T &var,
                 const F &lambda_fcn, T &result){
    
    for (int i=i_start; i<i_end; i++){
        for (int j=j_start; j<j_end; j++){
            lambda_fcn(i, j, var);
        }
    }
    
    result = var;
};  // end for_reduce


template <typename T, typename F>
void for_reduce (int i_start, int i_end,
                 int j_start, int j_end,
                 int k_start, int k_end,
                 T &var,
                 const F &lambda_fcn,  T &result){
    
    for (int i=i_start; i<i_end; i++){
        for (int j=j_start; j<j_end; j++){
            for (int k=k_start; k<k_end; k++){
                lambda_fcn(i,j,k,var);
            }
        }
    }
    
    result = var;
};  // end for_reduce


// -----------------------------------------
// MACROS used with both Kokkos and non-kokkos versions
// -----------------------------------------
// a macro to select the name of a macro based on the number of inputs
#define \
    GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, NAME,...) NAME


// -----------------------------------------
// MACROS for kokkos
// -----------------------------------------

#ifdef HAVE_KOKKOS

// CArray nested loop convention use Right, FArray use Left
#define LOOP_ORDER Kokkos::Iterate::Right


// the FOR_ALL loop
// 1D FOR loop has 4 inputs
#define \
    VARS4(i, x0, x1,fcn) \
    Kokkos::parallel_for( Kokkos::RangePolicy<> ( (x0), (x1)), \
                          KOKKOS_LAMBDA( const int (i) ){fcn} )
// 2D FOR loop has 7 inputs
#define \
    VARS7(i, x0, x1, j, y0, y1,fcn) \
    Kokkos::parallel_for( \
        Kokkos::MDRangePolicy< Kokkos::Rank<2,LOOP_ORDER,LOOP_ORDER> > ( {(x0), (y0)}, {(x1), (y1)} ), \
        KOKKOS_LAMBDA( const int (i), const int (j) ){fcn} )
// 3D FOR loop has 10 inputs
#define \
    VARS10(i, x0, x1, j, y0, y1, k, z0, z1, fcn) \
    Kokkos::parallel_for( \
         Kokkos::MDRangePolicy< Kokkos::Rank<3,LOOP_ORDER,LOOP_ORDER> > ( {(x0), (y0), (z0)}, {(x1), (y1), (z1)} ), \
         KOKKOS_LAMBDA( const int (i), const int (j), const int (k) ) {fcn} )

#define \
    FOR_ALL(...) \
    GET_MACRO(__VA_ARGS__, _12, _11, VARS10, _9, _8, VARS7, _6, _5, VARS4)(__VA_ARGS__)

// the FOR_REDUCE loop
#define \
    VARS6(i, x0, x1, var, fcn, result) \
    Kokkos::parallel_reduce( Kokkos::RangePolicy<> ( (x0), (x1) ),  \
                             KOKKOS_LAMBDA(const int (i), decltype(var) &(var)){fcn}, (result))

#define \
    VARS9(i, x0, x1, j, y0, y1, var, fcn, result) \
    Kokkos::parallel_reduce( \
        Kokkos::MDRangePolicy< Kokkos::Rank<2,LOOP_ORDER,LOOP_ORDER> > ( {(x0), (y0)}, {(x1), (y1)} ), \
        KOKKOS_LAMBDA( const int (i),const int (j), decltype(var) &(var) ){fcn}, \
           (result) )
#define \
    VARS12(i, x0, x1, j, y0, y1, k, z0, z1, var, fcn, result) \
    Kokkos::parallel_reduce( \
        Kokkos::MDRangePolicy< Kokkos::Rank<3,LOOP_ORDER,LOOP_ORDER> > ( {(x0), (y0), (z0)}, {(x1), (y1), (z1)} ), \
        KOKKOS_LAMBDA( const int (i), const int (j), const int (k), decltype(var) &(var) ){fcn}, \
            (result) )

#define \
    FOR_REDUCE(...) \
    GET_MACRO(__VA_ARGS__, VARS12, _11, _10, VARS9, _8, _7, VARS6)(__VA_ARGS__)
#endif


// -----------------------------------------
// MACROS for none kokkos loops
// -----------------------------------------

#ifndef HAVE_KOKKOS

// the FOR_ALL loop is chosen based on the number of inputs

// the FOR_ALL loop
// 1D FOR loop has 4 inputs
#define \
    VARS4(i, x0, x1, fcn) \
    for_all( (x0), (x1), \
             [&]( const int &(i) ){fcn} )
// 2D FOR loop has 7 inputs
#define \
    VARS7(i, x0, x1, j, y0, y1, fcn)  \
    for_all( (x0), (x1), (y0), (y1), \
             [&]( const int (i), const int (j) ){fcn} )
// 3D FOR loop has 10 inputs
#define \
    VARS10(i, x0, x1, j, y0, y1, k, z0, z1, fcn) \
    for_all( (x0), (x1), (y0), (y1), (z0), (z1), \
             [&]( const int (i), const int (j), const int (k) ) {fcn} )
#define \
    FOR_ALL(...) \
    GET_MACRO(__VA_ARGS__, _12, _11, VARS10, _9, _8, VARS7, _6, _5, VARS4)(__VA_ARGS__)

// the FOR_REDUCE loop ******* A work in progress ********
#define \
    VARS6(i, x0, x1, var, fcn, result) \
    for_reduce((x0), (x1), (var),  \
                [&]( const int (i), decltype(var) &(var) ){fcn}, \
                (result) )
#define \
    VARS9(i, x0, x1, j, y0, y1, var, fcn, result) \
    for_reduce((x0), (x1), (y0), (y1), (var),  \
               [&]( const int (i),const int (j), decltype(var) &(var) ){fcn}, \
               (result) )
#define \
    VARS12(i, x0, x1, j, y0, y1, k, z0, z1, var, fcn, result) \
    for_reduce((x0), (x1), (y0), (y1), (z0), (z1), (var),  \
               [&]( const int (i), const int (j), const int (k), decltype(var) &(var) ){fcn}, \
               (result) )

#define \
    FOR_REDUCE(...) \
    GET_MACRO(__VA_ARGS__, VARS12, _11, _10, VARS9, _8, _7, VARS6)(__VA_ARGS__)

#endif



