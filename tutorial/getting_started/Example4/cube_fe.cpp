#include <stdio.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include "matar.h"

// Required for MATAR data structures
using namespace mtr; 

int num_elements = 1;
int num_nodes_per_elem = 8;
int quadrature_order = 2; // Accurate to 2N-1
int dimensions = 3;
int num_gauss_in_elem = quadrature_order*dimensions;

CArrayDual<double> coordinates(num_elements, num_nodes_per_elem, dimensions);
CArrayDual<double> velocities(num_elements, num_nodes_per_elem, dimensions);
CArrayDual<double> forces(num_elements, num_nodes_per_elem, dimensions);


CArrayDevice<double> quadrature_points(num_elements, quadrature_order*dimensions);
CArrayDevice<double> quadrature_weights(num_elements, quadrature_order*dimensions);


double ref_node_array[8*3] = 
    {// listed as {Xi, Eta, Mu}
    // Bottom Nodes
    -1.0, -1.0, -1.0,// 0
    +1.0, -1.0, -1.0,// 1
    -1.0, +1.0, -1.0,// 2
    +1.0, +1.0, -1.0,// 3
    // Top Nodes
    -1.0, -1.0, +1.0,// 4
    +1.0, -1.0, +1.0,// 5
    -1.0, +1.0, +1.0,// 6
    +1.0, +1.0, +1.0 // 7
    };

ViewCArrayDevice<double> ref_nodes(ref_node_array, 8, 3);

CArrayDevice<double> basis_functions(num_elements, num_nodes_per_elem);



// main
int main(int argc, char* argv[])
{   
    MATAR_INITIALIZE(argc, argv);
    { // MATAR scope

    // Set the quadrature points and weights for the 1D case
    auto leg_points_1D = CArrayDevice<double> (quadrature_order);
    auto leg_weights_1D = CArrayDevice<double> (quadrature_order);
    
    // Initialize the quadrature points and weights for the 1D case
    RUN({ // NOTE: This runs in serial on the device, the KOKKOS_INLINE_FUNCTION declaration is required to call this function from the device
        legendre_nodes_1D(leg_points_1D, quadrature_order);
        legendre_weights_1D(leg_weights_1D, quadrature_order);
    })

    // Set the quadrature points and weights for each elements
    FOR_ALL(elem_id, 0, num_elements,
            i, 0, quadrature_order,
            j, 0, quadrature_order,
            k, 0, quadrature_order, {
            
            int gauss_lid = i + j*quadrature_order + k*quadrature_order*quadrature_order;
            
            quadrature_points(elem_id, gauss_lid, 0) = leg_points_1D(i);
            quadrature_points(elem_id, gauss_lid, 1) = leg_points_1D(j);
            quadrature_points(elem_id, gauss_lid, 2) = leg_points_1D(k);
            
            quadrature_weights(elem_id, gauss_lid) = leg_weights_1D(i)*leg_weights_1D(j)*leg_weights_1D(k); // NOTE: This is the product of the weights for the 1D case
    });

    initialize_coordinates(coordinates);





    }
    MATAR_FINALIZE();

    return 0;
}


KOKKOS_INLINE_FUNCTION
void legendre_nodes_1D(
    CArrayDevice<double> &leg_nodes_1D,
    const int &num){

    if (num == 1){
        leg_nodes_1D(0) = 0.0;
    }
    else if (num == 2){
        leg_nodes_1D(0) = -0.577350269189625764509148780501;
        leg_nodes_1D(1) =  0.577350269189625764509148780501;
    }
    else if (num == 3){
        leg_nodes_1D(0) = -0.774596669241483377035853079956;
        leg_nodes_1D(1) =  0.0;
        leg_nodes_1D(2) =  0.774596669241483377035853079956;
    }
    else{
        printf("Error: Invalid quadrature order\n");
    } // end if
}; // end of legendre_nodes_1D function

KOKKOS_INLINE_FUNCTION
void legendre_weights_1D(
    CArrayDevice<double> &leg_weights_1D,  // Legendre weights
    const int &num){                  // Interpolation order
    
    if (num == 1){
        leg_weights_1D(0) = 2.0;
    }
    else if (num == 2){
        leg_weights_1D(0) = 1.0;
        leg_weights_1D(1) = 1.0;
    }
    else if (num == 3){
        leg_weights_1D(0) = 0.555555555555555555555555555555555;
        leg_weights_1D(1) = 0.888888888888888888888888888888888;
        leg_weights_1D(2) = 0.555555555555555555555555555555555;
    }

    else{
        printf("Error: Invalid quadrature order\n");
    } // end if
} // end of legendre_weights_1D function

void initialize_coordinates(CArrayDual<double> &coordinates){
    
    FOR_ALL(elem_id, 0, num_elements,
            node_lid, 0, 8,
            j, 0, dimensions, {
            coordinates(elem_id, node_lid, j) = ref_nodes(node_lid, j);
    });

    FOR_ALL(i, 0, num_elements,
} // end of initialize_coordinates function


void basis_functions(CArrayDual<double> &basis_functions){
    
    FOR_ALL(elem_id, 0, num_elements,
            node_lid, 0, num_nodes_per_elem,{
        basis_functions(elem_id, node_lid) = 1.0/8.0
            * (1.0 + quadrature_points(elem_id, node_lid, 0)*ref_nodes(node_lid, 0))
            * (1.0 + quadrature_points(elem_id, node_lid, 1)*ref_nodes(node_lid, 1))
            * (1.0 + quadrature_points(elem_id, node_lid, 2)*ref_nodes(node_lid, 2));
    });
}





