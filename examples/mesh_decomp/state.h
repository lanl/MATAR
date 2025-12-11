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
#ifndef STATE_H
#define STATE_H

#include "matar.h"
// #include "mpi_type.h"

using namespace mtr;


// Possible node states, used to initialize node_t
enum class node_state
{
    coords,
    scalar_field,
    vector_field
};


/////////////////////////////////////////////////////////////////////////////
///
/// \struct node_t
///
/// \brief Stores state information associated with a node
///
/////////////////////////////////////////////////////////////////////////////
struct node_t
{

    // Replace with MPIDCArrayKokkos
    MPICArrayKokkos<double> coords;     ///< Nodal coordinates
    MPICArrayKokkos<double> coords_n0;  ///< Nodal coordinates at tn=0 of time integration
    
    MPICArrayKokkos<double> scalar_field; ///< Scalar field on a node
    MPICArrayKokkos<double> vector_field; ///< Vector field on a node


    // initialization method (num_nodes, num_dims, state to allocate)
    void initialize(size_t num_nodes, size_t num_dims, std::vector<node_state> node_states)
    {

        CommunicationPlan comm_plan;
        
        for (auto field : node_states){
            switch(field){
                case node_state::coords:
                    if (coords.size() == 0){
                        this->coords = MPICArrayKokkos<double>(num_nodes, num_dims, "node_coordinates");
                        this->coords.initialize_comm_plan(comm_plan);
                    }
                    if (coords_n0.size() == 0){
                        this->coords_n0 = MPICArrayKokkos<double>(num_nodes, num_dims, "node_coordinates_n0");
                        this->coords_n0.initialize_comm_plan(comm_plan);
                    }
                    break;
                case node_state::scalar_field:
                    if (scalar_field.size() == 0) this->scalar_field = MPICArrayKokkos<double>(num_nodes, "node_scalar_field");
                    this->scalar_field.initialize_comm_plan(comm_plan);
                    break;
                case node_state::vector_field:
                    if (vector_field.size() == 0) this->vector_field = MPICArrayKokkos<double>(num_nodes, num_dims, "node_vector_field");
                    this->vector_field.initialize_comm_plan(comm_plan);
                    break;
                default:
                    std::cout<<"Desired node state not understood in node_t initialize"<<std::endl;
                    throw std::runtime_error("**** Error in State Field Name ****");
            }
        }
    }; // end method
    
    // initialization method (num_nodes, num_dims, state to allocate)
    void initialize(size_t num_nodes, size_t num_dims, std::vector<node_state> node_states, CommunicationPlan& comm_plan)
    {
        for (auto field : node_states){
            switch(field){
                case node_state::coords:
                    if (coords.size() == 0){
                        this->coords = MPICArrayKokkos<double>(num_nodes, num_dims, "node_coordinates");
                        this->coords.initialize_comm_plan(comm_plan);
                    }
                    if (coords_n0.size() == 0){
                        this->coords_n0 = MPICArrayKokkos<double>(num_nodes, num_dims, "node_coordinates_n0");
                        this->coords_n0.initialize_comm_plan(comm_plan);
                    }
                    break;
                case node_state::scalar_field:
                    if (scalar_field.size() == 0) this->scalar_field = MPICArrayKokkos<double>(num_nodes, "node_scalar_field");
                    this->scalar_field.initialize_comm_plan(comm_plan);
                    break;
                case node_state::vector_field:
                    if (vector_field.size() == 0) this->vector_field = MPICArrayKokkos<double>(num_nodes, num_dims, "node_vector_field");
                    this->vector_field.initialize_comm_plan(comm_plan);
                    break;
                default:
                    std::cout<<"Desired node state not understood in node_t initialize"<<std::endl;
                    throw std::runtime_error("**** Error in State Field Name ****");
            }
        }
    }; // end method

}; // end node_t


// Possible gauss point states, used to initialize GaussPoint_t
enum class gauss_pt_state
{
    fields,
    fields_vec
};

/////////////////////////////////////////////////////////////////////////////
///
/// \struct GaussPoint_t
///
/// \brief Stores state information associated with the Gauss point
///
/////////////////////////////////////////////////////////////////////////////
struct GaussPoint_t
{

    MPICArrayKokkos<double> fields;
    MPICArrayKokkos<double> fields_vec;

    // initialization method (num_cells, num_dims)
    void initialize(size_t num_gauss_pnts, size_t num_dims, std::vector<gauss_pt_state> gauss_pt_states, CommunicationPlan& comm_plan)
    {

        for (auto field : gauss_pt_states){
            switch(field){
                case gauss_pt_state::fields:
                    //if (fields.size() == 0) this->fields = DCArrayKokkos<double>(num_gauss_pnts, "gauss_point_fields");
                    if (fields.size() == 0){
                        this->fields = MPICArrayKokkos<double>(num_gauss_pnts, "gauss_point_fields");
                        this->fields.initialize_comm_plan(comm_plan);
                    } 
                    break;
                case gauss_pt_state::fields_vec:
                    if (fields_vec.size() == 0){
                        this->fields_vec = MPICArrayKokkos<double>(num_gauss_pnts, num_dims, "gauss_point_fields_vec");
                        this->fields_vec.initialize_comm_plan(comm_plan);
                    } 
                    break;
                default:
                    std::cout<<"Desired gauss point state not understood in GaussPoint_t initialize"<<std::endl;
                    throw std::runtime_error("**** Error in State Field Name ****");
            }
        }
    }; // end method
};  // end GaussPoint_t



/////////////////////////////////////////////////////////////////////////////
///
/// \struct state_t
///
/// \brief Stores all state
///
/////////////////////////////////////////////////////////////////////////////
struct State_t
{
    // ---------------------------------------------------------------------
    //    state data on mesh declarations
    // ---------------------------------------------------------------------
    node_t node;              ///< access as node.coords(node_gid,dim)
    GaussPoint_t GaussPoints; ///< access as GaussPoints.vol(gauss_pt_gid)
    
}; // end state_t





#endif