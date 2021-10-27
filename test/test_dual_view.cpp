#include <stdio.h>

#include "matar.h"


void one_D_example();
void two_D_example();
void three_D_example();

int main() {

    Kokkos::initialize();
    { 

    // Run 1D example
    one_D_example();
 
    // Run 2D example
    two_D_example();
 
    // Run 3D example;
    three_D_example();

    } // end of kokkos scope
    Kokkos::finalize();
}


void one_D_example()
{
    printf("\n====================Running 1D example=======================\n");

    int nx = 2;

    // CPU arr
    int arr[nx];
    
    for (int i = 0; i < nx; i++){
        arr[i] = 1;
    }

    // Create A_1D
    auto A_1D = DViewCArrayKokkos <int> (&arr[0], nx);
    
    // Print device copy of A_1D
    printf("Printing device copy of dual view:\n");
    FOR_ALL(i, 0, nx, {
        printf("%d\n", A_1D(i));
    });
    Kokkos::fence();

    printf("Printing host copy of dual view:\n");
    for (int i = 0; i < nx; i++){
        printf("%d\n", A_1D.host(i));
    }

    // Manupulate A_1D on device
    FOR_ALL(i, 0, nx, {
        A_1D(i) = 2;
    });
    Kokkos::fence();

    // Update A_1D on host
    A_1D.update_host();

    // Print host copy of A_1D 
    printf("Printing host copy of dual view (Updated to 2 on device):\n");
    for (int i = 0; i < nx; i++){
        printf("%d\n", A_1D.host(i));
    }

    // Manupulate A_1D on host
    for (int i = 0; i < nx; i++){
        A_1D.host(i) = 3;
    }

    // Update A_1D on device
    A_1D.update_device();

    // Print device copy of A_1D 
    printf("Printing device copy of dual view (Updated to 3 on host):\n");
    FOR_ALL(i, 0, nx, {
        printf("%d\n", A_1D(i));
    });
    Kokkos::fence();
}



void two_D_example()
{
    printf("\n====================Running 2D example=======================\n");

    int nx = 2;
    int ny = 2;

    // CPU arr
    int arr[nx*ny];
    
    for (int i = 0; i < nx*ny; i++){
        arr[i] = 1;
    }

    // Create A_2D
    auto A_2D = DViewCArrayKokkos <int> (&arr[0], nx, ny);
    
    // Print device copy of A_2D
    printf("Printing device copy of dual view:\n");
    FOR_ALL(i, 0, nx,
            j, 0, ny, {
        printf("%d\n", A_2D(i,j));
    });
    Kokkos::fence();

    printf("Printing host copy of dual view:\n");
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            printf("%d\n", A_2D.host(i,j)); 
        }
    }

    // Manupulate A_2D on device
    FOR_ALL(i, 0, nx, 
            j, 0, ny, {
        A_2D(i,j) = 2;
    });
    Kokkos::fence();

    // Update A_2D on host
    A_2D.update_host();

    // Print host copy of A_2D 
    printf("Printing host copy of dual view (Updated to 2 on device):\n");
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            printf("%d\n", A_2D.host(i,j)); 
        }
    }

    // Manupulate A_2D on host
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            A_2D.host(i,j) = 3; 
        }
    }

    // Update A_2D on device
    A_2D.update_device();

    // Print device copy of A_2D 
    printf("Printing device copy of dual view (Updated to 3 on host):\n");
    FOR_ALL(i, 0, nx, 
            j, 0, ny, {
        printf("%d\n", A_2D(i,j));
    });
    Kokkos::fence();
}



void three_D_example()
{
    printf("\n====================Running 3D example=======================\n");

    int nx = 2;
    int ny = 2;
    int nz = 2;

    // CPU arr
    int arr[nx*ny*nz];
    
    for (int i = 0; i < nx*ny*nz; i++){
        arr[i] = 1;
    }

    // Create A_3D
    auto A_3D = DViewCArrayKokkos <int> (&arr[0], nx, ny, nz);
    
    // Print device copy of A_3D
    printf("Printing device copy of dual view:\n");
    FOR_ALL(i, 0, nx,
            j, 0, ny,
            k, 0, nz, {
        printf("%d\n", A_3D(i,j,k));
    });
    Kokkos::fence();

    printf("Printing host copy of dual view:\n");
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            for (int k = 0; k < nz; k++){
                printf("%d\n", A_3D.host(i,j,k));
            }
        }
    }

    // Manupulate A_3D on device
    FOR_ALL(i, 0, nx, 
            j, 0, ny,
            k, 0, nz, {
        A_3D(i,j,k) = 2;
    });
    Kokkos::fence();

    // Update A_3D on host
    A_3D.update_host();

    // Print host copy of A_3D 
    printf("Printing host copy of dual view (Updated to 2 on device):\n");
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            for (int k = 0; k < nz; k++){
                printf("%d\n", A_3D.host(i,j,k));
            }
        }
    }

    // Manupulate A_3D on host
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            for (int k = 0; k < nz; k++){
                A_3D.host(i,j,k) = 3;
            }
        }
    }

    // Update A_3D on device
    A_3D.update_device();

    // Print device copy of A_3D 
    printf("Printing device copy of dual view (Updated to 3 on host):\n");
    FOR_ALL(i, 0, nx, 
            j, 0, ny,
            k, 0, nz, {
        printf("%d\n", A_3D(i,j,k));
    });
    Kokkos::fence();
}
