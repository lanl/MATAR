
#include <stdio.h>
#include <iostream>
#include <matar.h>




// main
int main(){


    Kokkos::initialize();
{
    //Kokkos::View<int *> arr("ARR", 10);
    CArrayKokkos <int> arr(10);
    FOR_ALL (i, 0, 10, {
        arr(i) = 314;
    });

    //Kokkos::View<int **> arr_2D("ARR_2D", 10,10);
    CArrayKokkos <int> arr_2D(10,10);
    FOR_ALL (i, 0, 10, 
             j, 0, 10,{
        arr_2D(i,j) = 314;
    });

    //Kokkos::View<int ***> arr_3D("ARR_3D", 10,10,10);
    CArrayKokkos <int> arr_3D(10,10,10);
    FOR_ALL (i, 0, 10, 
             j, 0, 10,
             k, 0, 10,{
        arr_3D(i,j,k) = 314;
    });


    int loc_sum = 0;
    int result = 0;
    REDUCE_SUM(i, 0, 10,
               loc_sum, {
        loc_sum += arr(i)*arr(i);
    }, result);
    
    // testing
    loc_sum = 0;
    for (int i=0; i<10; i++){
        loc_sum += arr(i)*arr(i);
    }
    std::cout << "1D reduce : " << result << " vs. " << loc_sum << " \n";
    
    
    loc_sum = 0;
    result = 0;
    REDUCE_SUM(i, 0, 10,
               j, 0, 10,
               loc_sum, {
                   loc_sum += arr_2D(i,j)*arr_2D(i,j);
               }, result);
    
    // testing
    loc_sum = 0;
    for (int i=0; i<10; i++){
        for (int j=0; j<10; j++){
            loc_sum += arr_2D(i,j)*arr_2D(i,j);
        }
    }
    std::cout << "2D reduce : " << result << " vs. " << loc_sum << " \n";
    
    
    loc_sum = 0;
    result = 0;
    REDUCE_SUM(i, 0, 10,
               j, 0, 10,
               k, 0, 10,
               loc_sum, {
                   loc_sum += arr_3D(i,j,k)*arr_3D(i,j,k);
               }, result);
    
    // testing
    loc_sum = 0;
    for (int i=0; i<10; i++){
        for (int j=0; j<10; j++){
            for (int k=0; k<10; k++){
                loc_sum += arr_3D(i,j,k)*arr_3D(i,j,k);
            }
        }
    }
    std::cout << "3D reduce : " << result << " vs. " << loc_sum << " \n";
    
    result = 0;
    int loc_max = 2000;
    REDUCE_MAX(i, 0, 10,
               j, 0, 10,
               k, 0, 10,
               loc_max, {

                   if(loc_max < arr_3D(i,j,k)){
                       loc_max = arr_3D(i,j,k);
                   }
                   
               },
               result);
    
    std::cout << "3D reduce MAX : " << result << " \n";
    
    // verbose version
    double loc_max_value = 20000;
    double max_value = 20000;
    Kokkos::parallel_reduce(
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {10,10}),
                            KOKKOS_LAMBDA(const int i, const int j, double& loc_max_value)
                            {
                                if(arr_2D(i,j) > loc_max_value) loc_max_value = arr_2D(i,j);
                            },
                            Kokkos::Max<double>(max_value)
                            );
    std::cout << "2D reduce MAX kokkos verbose : " << max_value << " \n";
    
    result = 0;
    int loc_min = 2000;
    REDUCE_MIN(i, 0, 10,
               j, 0, 10,
               k, 0, 10,
               loc_min, {
                   
                   if(loc_min > arr_3D(i,j,k)){
                       loc_min = arr_3D(i,j,k); }
                   
               },
               result);
    
    std::cout << "3D reduce MIN : " << result << " \n";
    

    std::cout << "done" << std::endl;

}
    Kokkos::finalize();

    
    return 0;
}


