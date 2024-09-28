/**********************************************************************************************
 � 2020. Triad National Security, LLC. All rights reserved.
 This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
 National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
 Department of Energy/National Nuclear Security Administration. All rights in the program are
 reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
 Security Administration. The Government is granted for itself and others acting on its behalf a
 nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
 derivative works, distribute copies to the public, perform publicly and display publicly, and
 to permit others to do so.
 This program is open source under the BSD-3 License.
 Redistribution and use in source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:
 1.  Redistributions of source code must retain the above copyright notice, this list of
 conditions and the following disclaimer.
 2.  Redistributions in binary form must reproduce the above copyright notice, this list of
 conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.
 3.  Neither the name of the copyright holder nor the names of its contributors may be used
 to endorse or promote products derived from this software without specific prior
 written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************************************/
#include <stdio.h>
#include <array>
#include <vector>
#include <chrono>
#include <math.h>

#include "matar.h"

using namespace mtr; // matar namespace



// =================================================================
// Artificial Neural Network (ANN)
//
// For a single layer, we have x_i inputs with weights_{ij}, 
// creating y_j outputs.  We have
//     y_j = Fcn(b_j) = Fcn( Sum_i {x_i w_{ij}} )
// where the activation function Fcn is applied to b_j, creating 
// outputs y_j. For multiple layers, we have
//      b_j^l = Sum_i (x_i^{l-1} w_{ij}^l)
// where l is a layer, and as before, an activation function is  
// applied to b_j^l, creating outputs y_j^l.
// 
// =================================================================


// =================================================================
//
// Number of nodes in each layer including inputs and outputs
//
// =================================================================
std::vector <size_t> num_nodes_in_layer = {9, 50, 100, 300, 200, 100, 20, 6};
// {64000, 30000, 8000, 4000, 100, 25, 6}



// =================================================================
//
// data types and classes
//
// =================================================================

// array of ANN structs
struct ANNLayer_t{

    DCArrayKokkos <float> outputs;  // dims = [layer]
    DFArrayKokkos <float> weights;  // dims = [layer-1, layer]

}; // end struct



// =================================================================
//
// functions
//
// =================================================================
void vec_mat_multiply(DCArrayKokkos <float> &inputs,
                      DCArrayKokkos <float> &outputs, 
                      DFArrayKokkos <float> &matrix){
    
    const size_t num_i = inputs.size();
    const size_t num_j = outputs.size();

    using team_t = typename Kokkos::TeamPolicy<>::member_type;
    Kokkos::parallel_for ("MatVec", Kokkos::TeamPolicy<> (num_j, Kokkos::AUTO),
                 KOKKOS_LAMBDA (const team_t& team_h) {

        float sum = 0;
        int j = team_h.league_rank();
        Kokkos::parallel_reduce (Kokkos::TeamThreadRange (team_h, num_i),
                        [&] (int i, float& lsum) {
            lsum += inputs(i)*matrix(i,j);
        }, sum); // end parallel reduce

        outputs(j) = sum; 

    }); // end parallel for
    
    return;

}; // end function



void forward_propagate_layer(DCArrayKokkos <float> &inputs,
                             DCArrayKokkos <float> &outputs, 
                             DFArrayKokkos <float> &weights){
    
    const size_t num_i = inputs.size();
    const size_t num_j = outputs.size();

    
    FOR_ALL(j, 0, num_j,{


    	//printf("thread = %d \n", omp_get_thread_num());

            float value = 0.0;
            for(int i=0; i<num_i; i++){
                // b_j = Sum_i {x_i w_{ij}}
                value += inputs(i)*weights(i,j);
            } // end for

            // apply activation function, sigmoid on a float, y_j = Fcn(b_j)
            outputs(j) = 1.0/(1.0 + exp(-value));  // exp2f doesn't work with CUDA

        }); // end parallel for
    
    /*
    
    using team_t = typename Kokkos::TeamPolicy<>::member_type;
    Kokkos::parallel_for ("MatVec", Kokkos::TeamPolicy<> (num_j, Kokkos::AUTO),
                 KOKKOS_LAMBDA (const team_t& team_h) {

        float sum = 0;
        int j = team_h.league_rank();
        Kokkos::parallel_reduce (Kokkos::TeamThreadRange (team_h, num_i),
                        [&] (int i, float& lsum) {
            lsum += inputs(i)*weights(i,j);
        }, sum); // end parallel reduce

        outputs(j) = 1.0/(1.0 + exp(-sum)); 

    }); // end parallel for
    
    */
    
    return;

}; // end function



void set_weights(const DFArrayKokkos <float> &weights){

    const size_t num_i = weights.dims(0);
    const size_t num_j = weights.dims(1);
    
	FOR_ALL(i,0,num_i,
	        j,0,num_j, {
		    
		    weights(i,j) = 1.0;
	}); // end parallel for

}; // end function


// =================================================================
//
// Main function
//
// =================================================================
int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {

        // =================================================================
        // allocate arrays
        // =================================================================

        // note: the num_nodes_in_layer has the inputs into the ANN, so subtract 1 for the layers
        size_t num_layers = num_nodes_in_layer.size()-1;  

        CMatrix <ANNLayer_t> ANNLayers(num_layers); // starts at 1 and goes to num_layers

        // input and ouput values to ANN
        DCArrayKokkos <float> inputs(num_nodes_in_layer[0]);


        // set the strides
        // layer 0 are the inputs to the ANN
        // layer n-1 are the outputs from the ANN
        for (size_t layer=1; layer<=num_layers; layer++){

            // dimensions
            size_t num_i = num_nodes_in_layer[layer-1];
            size_t num_j = num_nodes_in_layer[layer];

            // allocate the weights in this layer
            ANNLayers(layer).weights = DFArrayKokkos <float> (num_i, num_j); 
            ANNLayers(layer).outputs = DCArrayKokkos <float> (num_j);

        } // end for


        // =================================================================
        // set weights and inputs
        // =================================================================
        
        // inputs to ANN
        for (size_t i=0; i<num_nodes_in_layer[0]; i++) {
            inputs.host(i) = 1.0;
        }
        inputs.update_device();  // copy inputs to device

        // weights of the ANN
        for (size_t layer=1; layer<=num_layers; layer++){

            // dimensions
            size_t num_i = num_nodes_in_layer[layer-1];
            size_t num_j = num_nodes_in_layer[layer];


            set_weights(ANNLayers(layer).weights);

        } // end for over layers



        // =================================================================
        // Testing vec matrix multiply
        // =================================================================        
        vec_mat_multiply(inputs,
                         ANNLayers(1).outputs,
                         ANNLayers(1).weights); 
        
        std::cout << "vec mat multiply test: \n";
        const size_t dims_in = ANNLayers(1).weights.dims(0);
        FOR_ALL(i,0,num_nodes_in_layer[1], {
            if(fabs(ANNLayers(1).outputs(i) - dims_in)>= 1e-15){
                printf("error in vec mat multiply test \n");
            }
        });



        // =================================================================
        // Use the ANN
        // =================================================================

        auto time_1 = std::chrono::high_resolution_clock::now();

        // forward propogate

        // layer 1, hidden layer 0, uses the inputs as the input values
        forward_propagate_layer(inputs,
                                ANNLayers(1).outputs,
                                ANNLayers(1).weights); 

        // layer 2 through n-1, layer n-1 goes to the output
        for (size_t layer=2; layer<=num_layers; layer++){

            // go through this layer, the fcn takes(inputs, outputs, weights)
            forward_propagate_layer(ANNLayers(layer-1).outputs, 
                                    ANNLayers(layer).outputs,
                                    ANNLayers(layer).weights); 
        } // end for

        auto time_2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration <float, std::milli> ms = time_2 - time_1;
        std::cout << "runtime of ANN test = " << ms.count() << "ms\n\n";


        // =================================================================
        // Copy values to host
        // =================================================================
        ANNLayers(num_layers).outputs.update_device();
        
        std::cout << "output values: \n";
        for (size_t val=0; val<num_nodes_in_layer[num_layers]; val++){
            std::cout << " " << ANNLayers(num_layers).outputs.host(val) << std::endl;
        } // end for
 
    } // end of kokkos scope

    Kokkos::finalize();



    printf("\nfinished\n\n");

    return 0;
}


