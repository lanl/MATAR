#include <stdio.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include "matar.h"

// Required for MATAR data structures
using namespace mtr; 

int num_elements = 1;
int num_nodes_per_elem = 8; // 8 NODE HEX
int quadrature_order = 2; // Accurate to 2N-1
int dimensions = 3;
int num_gauss_in_elem = quadrature_order*dimensions;

CArrayDual<double> initial_coordinates(num_elements, num_nodes_per_elem, dimensions);
CArrayDual<double> updated_coordinates(num_elements, num_nodes_per_elem, dimensions);
CArrayDevice<double> quadrature_points(num_elements, quadrature_order*dimensions);
CArrayDevice<double> quadrature_weights(num_elements, quadrature_order*dimensions);


CArrayDual<double> deformation_gradient(num_elements, dimensions, dimensions);

// Element stiffness matrix (Ke)
CArrayDevice<double> Ke(num_elements, num_nodes_per_elem*dimensions, num_nodes_per_elem*dimensions);
// Element force vector (fe)
CArrayDevice<double> fe(num_elements, num_nodes_per_elem*dimensions);
// Solution displacement vector (ue)
CArrayDevice<double> ue(num_elements, num_nodes_per_elem*dimensions);





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

    init_data();

    double mu = 80e9;      // Shear modulus
    double lambda = 120e9; // First Lamé parameter (unused here)

    




    }
    MATAR_FINALIZE();

    return 0;
}


void init_data(){

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

    FOR_ALL(elem_id, 0, num_elements,
            node_lid, 0, 8,
            j, 0, dimensions, {
            initial_coordinates(elem_id, node_lid, j) = ref_nodes(node_lid, j);
    });


    updated_coordinates.set_values(0.0);
    Ke.set_values(0.0);
    fe.set_values(0.0);
    ue.set_values(0.0);


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

void basis_functions(CArrayDual<double> &basis_functions){
    
    FOR_ALL(elem_id, 0, num_elements,
            node_lid, 0, num_nodes_per_elem,{
        basis_functions(elem_id, node_lid) = 1.0/8.0
            * (1.0 + quadrature_points(elem_id, node_lid, 0)*ref_nodes(node_lid, 0))
            * (1.0 + quadrature_points(elem_id, node_lid, 1)*ref_nodes(node_lid, 1))
            * (1.0 + quadrature_points(elem_id, node_lid, 2)*ref_nodes(node_lid, 2));
    });
}

// Applies a given 3x3 deformation gradient tensor F to the initial nodal coordinates.
// F: 3x3 deformation gradient
void apply_deformation(const CArrayDual<double> &F) {
    
    FOR_ALL(elem_id, 0, num_elements,
            node_lid, 0, num_nodes_per_elem,
            j, 0, dimensions, {
                
                updated_coordinates(elem_id, node_lid, j) = 0.0;
                
                for (int k = 0; k < dimensions; ++k){
                    updated_coordinates(elem_id, node_lid, j) += F(elem_id, j, k) * initial_coordinates(elem_id, node_lid, k);
                }
    });
}


// Computes the derivatives of the shape functions for an 8-node hexahedral element
// with respect to the local coordinates xi, eta, zeta.
// xi, eta, zeta: local coordinates in the reference element [-1,1]
// dN_dxi: output array containing shape function derivatives [8 nodes][3 components]
void shape_function_derivatives(double xi, double eta, double zeta, double dN_dxi[8][3]) {
    for (int i = 0; i < 8; ++i) {
        double sx = (i & 1) ? 1 : -1;
        double sy = (i & 2) ? 1 : -1;
        double sz = (i & 4) ? 1 : -1;

        dN_dxi[i][0] = 0.125 * sx * (1 + sy * eta) * (1 + sz * zeta);
        dN_dxi[i][1] = 0.125 * sy * (1 + sx * xi)  * (1 + sz * zeta);
        dN_dxi[i][2] = 0.125 * sz * (1 + sx * xi)  * (1 + sy * eta);
    }
}


void shape_function_derivatives(CArrayDevice<double> &ref_position, CArrayDual<double> &dN_dxi) {
    for (int i = 0; i < 8; ++i) {
        double sx = (i & 1) ? 1 : -1;
        double sy = (i & 2) ? 1 : -1;
        double sz = (i & 4) ? 1 : -1;

        dN_dxi(i,0) = 0.125 * sx * (1 + sy * ref_position(1)) * (1 + sz * ref_position(2));
        dN_dxi(i,1) = 0.125 * sy * (1 + sx * ref_position(0))  * (1 + sz * ref_position(2));
        dN_dxi(i,2) = 0.125 * sz * (1 + sx * ref_position(0))  * (1 + sy * ref_position(1));
    }
}

/**************************************************************************************//**
*  mat_inverse is a light weight function for inverting a 3X3 matrix. 
*****************************************************************************************/
void mat_inverse(
    ViewCArray <double> &mat_inv,
    ViewCArray <double> &matrix){

    // computes the inverse of a matrix m
    double det_a = matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(2, 1) * matrix(1, 2)) -
                   matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
                   matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));

    if (fabs(det_a) < 1e-12) return false;

    double invdet = 1.0 / det_a;

    mat_inv(0, 0) = (matrix(1, 1) * matrix(2, 2) - matrix(2, 1) * matrix(1, 2)) * invdet;
    mat_inv(0, 1) = (matrix(0, 2) * matrix(2, 1) - matrix(0, 1) * matrix(2, 2)) * invdet;
    mat_inv(0, 2) = (matrix(0, 1) * matrix(1, 2) - matrix(0, 2) * matrix(1, 1)) * invdet;
    mat_inv(1, 0) = (matrix(1, 2) * matrix(2, 0) - matrix(1, 0) * matrix(2, 2)) * invdet;
    mat_inv(1, 1) = (matrix(0, 0) * matrix(2, 2) - matrix(0, 2) * matrix(2, 0)) * invdet;
    mat_inv(1, 2) = (matrix(1, 0) * matrix(0, 2) - matrix(0, 0) * matrix(1, 2)) * invdet;
    mat_inv(2, 0) = (matrix(1, 0) * matrix(2, 1) - matrix(2, 0) * matrix(1, 1)) * invdet;
    mat_inv(2, 1) = (matrix(2, 0) * matrix(0, 1) - matrix(0, 0) * matrix(2, 1)) * invdet;
    mat_inv(2, 2) = (matrix(0, 0) * matrix(1, 1) - matrix(1, 0) * matrix(0, 1)) * invdet;

    return true;
}