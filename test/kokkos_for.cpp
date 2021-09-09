
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
    FOR_REDUCE(i, 0, 10,
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
    FOR_REDUCE(i, 0, 10,
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
    FOR_REDUCE(i, 0, 10,
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
    
    

    std::cout << "done" << std::endl;

}
    Kokkos::finalize();

    
    return 0;
}


