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
#include <chrono>
#include <math.h>

#include "matar.h"

using namespace mtr; // matar namespace


// =================================================================
//
// Number of nodes in each layer including input and output layers
//
// =================================================================
std::array <size_t,5> num_nodes_in_layer = {1, 10, 20, 10, 1};



// =================================================================
//
// data types and classes
//
// =================================================================

// array of ANN structs
struct ANNLayer_t{

    DCArrayKokkos <float> outputs;   // dims = [layer]
    DCArrayKokkos <float> weights;  // dims = [layer-1, layer]

}; // end struct



// =================================================================
//
// functions
//
// =================================================================
void forward_propagate_layer(DCArrayKokkos <float> inputs,
                             DCArrayKokkos <float> outputs, 
                             CArrayKokkos <float> weights){
    
    size_t i_dim = inputs.size();
    size_t j_dim = outputs.size();
    FOR_ALL(j, 0, j_dim,{

            float value = 0.0;
            for(int i=0; i<i_dim; i++){
                // b_j = Sum_i {x_i w_{ij}}
                value += weigths(i,j)*inputs(i);
            } // end for

            // apply activation function, sigmoid on a float, y_j = Fcn(b_j)
            outputs(j) = 1.0/(1.0 + exp2f(-value));

        }); // end parallel for

    return 0;

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

        //------------------
        // test of ANN
        //
        // For a single layer, we have x_i inputs and weights_{ij}, 
        // creating y_j outputs.
        //      b_j = Sum_i {x_i w_{ij}}
        // An activation function is applied to b_j, creating output y_j. 
        // For multiple layers, we have
        //      b_j^l = Sum_i (x_i^{l-1} w_{ij}^l)
        // where l is a layer and an activation function is applied to 
        // b_j^l, creating an output y_j^l.
        // 
        //------------------


        // =================================================================
        // allocate arrays
        // =================================================================

        num_layers = num_nodes_in_layer.size();       // includes inputs into ANN
        CMatrix <ANNLayer_t> ANNLayers(num_layers-1); // starts at 1 and goes to n

        // input and ouput values to ANN
        DCArrayKokkos <float> inputs(num_nodes_in_layer[0]);


        // set the strides
        // layer 0 are the inputs to the ANN
        // layer n-1 are the outputs from the ANN
        for (size_t layer=1; layer<num_layers; layer++){

            // dimensions
            size_t num_i = num_nodes_in_layer(layer-1);
            size_t num_j = num_nodes_in_layer(layer);

            // allocate the weights in this layer
            ANNLayers(layer).weights = DCArrayKokkos <float> (num_i, num_j); 
            ANNLayers(layer).outputs = DCArrayKokkos <float> (num_j);

        } // end for


        // =================================================================
        // set weights and inputs
        // =================================================================
        
        // inputs to ANN
        num_i = num_nodes_in_layer[0];
        for (size_t i=0; i<num_i; i++) {
            inputs.host(i) = 1.0;
        }

        // weights of the ANN
        for (size_t layer=1; layer<num_layers; layer++){

            // dimensions
            size_t num_i = num_nodes_in_layer[layer-1];
            size_t num_j = num_nodes_in_layer[layer];

            // set the weights in this layer of the ANN
            for (size_t i=0; i<num_i; i++) {
                for (size_t j=0; j<num_j; j++){
                    ANNLayers(layer).weights.host(i,j) = 1.0;
                } // end for j
            }  // end for
            ANNLayers(layer).weights.update_device();  // copy weights to device

        } // end for over layers


        // =================================================================
        // Use the ANN
        // =================================================================

        auto time_1 = std::chrono::high_resolution_clock::now();

        // forward propogate

        // layer 1, hidden layer 0, uses the inputs
        forward_propagate_layer(inputs,
                                ANNLayers(1).outputs,
                                ANNLayers(1).weights); 

        // layer 2 through n
        for (size_t layer=2; layer<num_layers; layer++){

            // dimensions
            size_t num_nodes_i_minus = num_nodes_in_layer[layer-1];
            size_t num_nodes_i = num_nodes_in_layer[layer];

            // go through this layer, the fcn takes(inputs, outputs, weights)
            forward_propagate_layer(ANNLayers(layer-1).outputs, 
                                    ANNLayers(layer).outputs
                                    ANNLayers(layer).weights); 
        } // end for

        auto time_2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration <float, std::milli> ms = time_2 - time_1;
        std::cout << "runtime of ann test = " << ms.count() << "ms\n";


        // =================================================================
        // Copy values to host
        // =================================================================
        ANNLayers(num_layers).outputs.update_device();

 
    } // end of kokkos scope

    Kokkos::finalize();

    printf("\nfinished\n\n");

    return 0;
}


