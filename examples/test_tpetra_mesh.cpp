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
#include <iostream>
#include <string>
#include <sstream>

#include "matar.h"
#include "Kokkos_DualView.hpp"

using namespace mtr; // matar namespace

void setup_maps();
void read_mesh_vtk(const char* MESH);

int main(int argc, char* argv[])
{   
    MPI_Init(&argc, &argv);
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    Kokkos::initialize();
    {

        // Run TpetraFArray 7D example
        read_mesh_vtk(argv[0]);
    } // end of kokkos scope
    Kokkos::finalize();
    MPI_Barrier(MPI_COMM_WORLD);
    if(process_rank==0)
        printf("\nfinished\n\n");
    MPI_Finalize();
}

void setup_maps()
{
    
}

/* ----------------------------------------------------------------------
   Read VTK format mesh file
------------------------------------------------------------------------- */

void read_mesh_vtk(const char* MESH)
{
    char ch;
    std::string skip_line, read_line, substring;
    std::stringstream line_parse;

    int num_dim = 3;
    int local_node_index, current_column_index;
    int buffer_loop, buffer_iteration, buffer_iterations, dof_limit, scan_loop;
    int negative_index_found = 0;
    int global_negative_index_found = 0;

    size_t read_index_start, node_rid, elem_gid;
    size_t strain_count;
    size_t nlocal_nodes;
    size_t buffer_nlines = 100000;

    GO     node_gid;
    real_t dof_value;
    real_t unit_scaling                  = 1.0;
    bool   zero_index_base               = false;

    CArrayKokkos<char, array_layout, HostSpace, memory_traits> read_buffer;

    //corresponding MPI rank for this process
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // read the mesh
    // --- Read the number of nodes in the mesh --- //
    num_nodes = 0;
    if (process_rank == 0)
    {
        std::cout << " NUM DIM is " << num_dim << std::endl;
        in = new std::ifstream();
        in->open(MESH);
        bool found = false;

        while (found == false&&in->good()) {
            std::getline(*in, read_line);
            line_parse.str("");
            line_parse.clear();
            line_parse << read_line;
            line_parse >> substring;

            // looking for the following text:
            //      POINTS %d float
            if (substring == "POINTS")
            {
                line_parse >> num_nodes;
                std::cout << "declared node count: " << num_nodes << std::endl;
                if (num_nodes <= 0)
                {
                    throw std::runtime_error("ERROR, NO NODES IN MESH");
                }
                found = true;
            } // end if
        } // end while

        if (!found){
            throw std::runtime_error("ERROR: Failed to find POINTS");
        } // end if

    } // end if(process_rank==0)

    // broadcast number of nodes
    MPI_Bcast(&num_nodes, 1, MPI_LONG_LONG_INT, 0, world);

    // construct distributed storage for node coordinates now that we know the global number of nodes
    // the default map of this array assigns an ordered contiguous subset of the global IDs to each process
    TpetraDFArray<double> node_coords_distributed(num_nodes, num_dim);
    // node_coords_distributed.pmap.print();

    //map of the distributed node coordinates vector
    TpetraPartitionMap<> map = node_coords_distributed.pmap;

    // set the local number of nodes on this process
    nlocal_nodes = node_coords_distributed.dims(0);

    //std::cout << "Num nodes assigned to task " << process_rank << " = " << nlocal_nodes << std::endl;

    // read the initial mesh coordinates
    /*only task 0 reads in nodes and elements from the input file
    stores node data in a buffer and communicates once the buffer cap is reached
    or the data ends*/

    words_per_line = num_dim;
    if(num_dim==2)
        elem_words_per_line = 4;
    else if(num_dim==3)
        elem_words_per_line = 8;

    // allocate read buffer
    read_buffer = CArrayKokkos<char, array_layout, HostSpace, memory_traits>(buffer_nlines, words_per_line, MAX_WORD);

    dof_limit = num_nodes;
    buffer_iterations = dof_limit / buffer_nlines;
    if (dof_limit % buffer_nlines != 0)
    {
        buffer_iterations++;
    }

    // read coords
    read_index_start = 0;
    for (buffer_iteration = 0; buffer_iteration < buffer_iterations; buffer_iteration++)
    {
        // pack buffer on rank 0
        if (process_rank == 0 && buffer_iteration < buffer_iterations - 1)
        {
            for (buffer_loop = 0; buffer_loop < buffer_nlines; buffer_loop++)
            {
                getline(*in, read_line);
                line_parse.clear();
                line_parse.str(read_line);

                for (int iword = 0; iword < words_per_line; iword++)
                {
                    // read portions of the line into the substring variable
                    line_parse >> substring;
                    // debug print
                    // std::cout<<" "<< substring <<std::endl;
                    // assign the substring variable as a word of the read buffer
                    strcpy(&read_buffer(buffer_loop, iword, 0), substring.c_str());
                }
            }
        }
        else if (process_rank == 0)
        {
            buffer_loop = 0;
            while (buffer_iteration * buffer_nlines + buffer_loop < num_nodes) {
                getline(*in, read_line);
                line_parse.clear();
                line_parse.str(read_line);
                for (int iword = 0; iword < words_per_line; iword++)
                {
                    // read portions of the line into the substring variable
                    line_parse >> substring;
                    // debug print
                    // std::cout<<" "<< substring <<std::endl;
                    // assign the substring variable as a word of the read buffer
                    strcpy(&read_buffer(buffer_loop, iword, 0), substring.c_str());
                }
                buffer_loop++;
            }
        }

        // broadcast buffer to all ranks; each rank will determine which nodes in the buffer belong
        MPI_Bcast(read_buffer.pointer(), buffer_nlines * words_per_line * MAX_WORD, MPI_CHAR, 0, world);
        // broadcast how many nodes were read into this buffer iteration
        MPI_Bcast(&buffer_loop, 1, MPI_INT, 0, world);

        // debug_print
        // std::cout << "NODE BUFFER LOOP IS: " << buffer_loop << std::endl;
        // for(int iprint=0; iprint < buffer_loop; iprint++)
        // std::cout<<"buffer packing: " << std::string(&read_buffer(iprint,0,0)) << std::endl;
        // return;

        // determine which data to store in the swage mesh members (the local node data)
        // loop through read buffer
        for (scan_loop = 0; scan_loop < buffer_loop; scan_loop++)
        {
            // set global node id (ensight specific order)
            node_gid = read_index_start + scan_loop;
            // let map decide if this node id belongs locally; if yes store data
            if (map.isProcessGlobalIndex(node_gid))
            {
                // set local node index in this mpi rank
                node_rid = map.getLocalIndex(node_gid);
                // extract nodal position from the read buffer
                // for tecplot format this is the three coords in the same line
                dof_value = atof(&read_buffer(scan_loop, 0, 0));
                node_coords_distributed.host(node_rid, 0) = dof_value * unit_scaling;
                dof_value = atof(&read_buffer(scan_loop, 1, 0));
                node_coords_distributed.host(node_rid, 1) = dof_value * unit_scaling;
                if (num_dim == 3)
                {
                    dof_value = atof(&read_buffer(scan_loop, 2, 0));
                    node_coords_distributed.host(node_rid, 2) = dof_value * unit_scaling;
                }
            }
        }
        read_index_start += buffer_nlines;
    }
    // repartition node distribution
    node_coords_distributed.update_device();
    node_coords_distributed.repartition_vector();
    //reset our local map variable to the repartitioned map
    TpetraPartitionMap<> map = node_coords_distributed.pmap;

    // synchronize device data


    // check that local assignments match global total

    // read in element info (ensight file format is organized in element type sections)
    // loop over this later for several element type sections

    num_elem  = 0;
    rnum_elem = 0;
    CArrayKokkos<int, array_layout, HostSpace, memory_traits> node_store(elem_words_per_line);

    // --- read the number of cells in the mesh ---
    // --- Read the number of vertices in the mesh --- //
    if (process_rank == 0)
    {
        bool found = false;
        while (found == false&&in->good()) {
            std::getline(*in, read_line);
            line_parse.str("");
            line_parse.clear();
            line_parse << read_line;
            line_parse >> substring;

            // looking for the following text:
            //      CELLS num_cells size
            if (substring == "CELLS")
            {
                line_parse >> num_elem;
                std::cout << "declared element count: " << num_elem << std::endl;
                if (num_elem <= 0)
                {
                    throw std::runtime_error("ERROR, NO ELEMENTS IN MESH");
                }
                found = true;
            } // end if
        } // end while

        if (!found){
            throw std::runtime_error("ERROR: Failed to find CELLS");
        } // end if
    } // end if(process_rank==0)

    // broadcast number of elements
    MPI_Bcast(&num_elem, 1, MPI_LONG_LONG_INT, 0, world);

    if (process_rank == 0)
    {
        std::cout << "before mesh initialization" << std::endl;
    }

    // read in element connectivity
    // we're gonna reallocate for the words per line expected for the element connectivity
    read_buffer = CArrayKokkos<char, array_layout, HostSpace, memory_traits>(buffer_nlines, elem_words_per_line, MAX_WORD);

    // calculate buffer iterations to read number of lines
    buffer_iterations = num_elem / buffer_nlines;
    int assign_flag;

    // dynamic buffer used to store elements before we know how many this rank needs
    std::vector<size_t> element_temp(buffer_nlines * elem_words_per_line);
    std::vector<size_t> global_indices_temp(buffer_nlines);
    size_t buffer_max = buffer_nlines * elem_words_per_line;
    size_t indices_buffer_max = buffer_nlines;

    if (num_elem % buffer_nlines != 0)
    {
        buffer_iterations++;
    }
    read_index_start = 0;
    // std::cout << "ELEMENT BUFFER ITERATIONS: " << buffer_iterations << std::endl;
    rnum_elem = 0;
    for (buffer_iteration = 0; buffer_iteration < buffer_iterations; buffer_iteration++)
    {
        // pack buffer on rank 0
        if (process_rank == 0 && buffer_iteration < buffer_iterations - 1)
        {
            for (buffer_loop = 0; buffer_loop < buffer_nlines; buffer_loop++)
            {
                getline(*in, read_line);
                line_parse.clear();
                line_parse.str(read_line);
                // disregard node count line since we're using one element type per mesh
                line_parse >> substring;
                for (int iword = 0; iword < elem_words_per_line; iword++)
                {
                    // read portions of the line into the substring variable
                    line_parse >> substring;
                    // debug print
                    // std::cout<<" "<< substring;
                    // assign the substring variable as a word of the read buffer
                    strcpy(&read_buffer(buffer_loop, iword, 0), substring.c_str());
                }
                // std::cout <<std::endl;
            }
        }
        else if (process_rank == 0)
        {
            buffer_loop = 0;
            while (buffer_iteration * buffer_nlines + buffer_loop < num_elem) {
                getline(*in, read_line);
                line_parse.clear();
                line_parse.str(read_line);
                line_parse >> substring;
                for (int iword = 0; iword < elem_words_per_line; iword++)
                {
                    // read portions of the line into the substring variable
                    line_parse >> substring;
                    // debug print
                    // std::cout<<" "<< substring;
                    // assign the substring variable as a word of the read buffer
                    strcpy(&read_buffer(buffer_loop, iword, 0), substring.c_str());
                }
                // std::cout <<std::endl;
                buffer_loop++;
                // std::cout<<" "<< node_coords_distributed(node_gid, 0)<<std::endl;
            }
        }

        // broadcast buffer to all ranks; each rank will determine which nodes in the buffer belong
        MPI_Bcast(read_buffer.pointer(), buffer_nlines * elem_words_per_line * MAX_WORD, MPI_CHAR, 0, world);
        // broadcast how many nodes were read into this buffer iteration
        MPI_Bcast(&buffer_loop, 1, MPI_INT, 0, world);

        // store element connectivity that belongs to this rank
        // loop through read buffer
        for (scan_loop = 0; scan_loop < buffer_loop; scan_loop++)
        {
            // set global node id (ensight specific order)
            elem_gid = read_index_start + scan_loop;
            // add this element to the local list if any of its nodes belong to this rank according to the map
            // get list of nodes for each element line and check if they belong to the map
            assign_flag = 0;
            for (int inode = 0; inode < elem_words_per_line; inode++)
            {
                // as we loop through the nodes belonging to this element we store them
                // if any of these nodes belongs to this rank this list is used to store the element locally
                node_gid = atoi(&read_buffer(scan_loop, inode, 0));
                if (zero_index_base)
                {
                    node_store(inode) = node_gid; // subtract 1 since file index start is 1 but code expects 0
                }
                else
                {
                    node_store(inode) = node_gid - 1; // subtract 1 since file index start is 1 but code expects 0
                }
                if (node_store(inode) < 0)
                {
                    negative_index_found = 1;
                }
                // first we add the elements to a dynamically allocated list
                if (zero_index_base)
                {
                    if (map.isProcessGlobalIndex(node_gid) && !assign_flag)
                    {
                        assign_flag = 1;
                        rnum_elem++;
                    }
                }
                else
                {
                    if (map.isProcessGlobalIndex(node_gid - 1) && !assign_flag)
                    {
                        assign_flag = 1;
                        rnum_elem++;
                    }
                }
            }

            if (assign_flag)
            {
                for (int inode = 0; inode < elem_words_per_line; inode++)
                {
                    if ((rnum_elem - 1) * elem_words_per_line + inode >= buffer_max)
                    {
                        element_temp.resize((rnum_elem - 1) * elem_words_per_line + inode + buffer_nlines * elem_words_per_line);
                        buffer_max = (rnum_elem - 1) * elem_words_per_line + inode + buffer_nlines * elem_words_per_line;
                    }
                    element_temp[(rnum_elem - 1) * elem_words_per_line + inode] = node_store(inode);
                    // std::cout << "VECTOR STORAGE FOR ELEM " << rnum_elem << " ON TASK " << process_rank << " NODE " << inode+1 << " IS " << node_store(inode) + 1 << std::endl;
                }
                // assign global element id to temporary list
                if (rnum_elem - 1 >= indices_buffer_max)
                {
                    global_indices_temp.resize(rnum_elem - 1 + buffer_nlines);
                    indices_buffer_max = rnum_elem - 1 + buffer_nlines;
                }
                global_indices_temp[rnum_elem - 1] = elem_gid;
            }
        }
        read_index_start += buffer_nlines;
    }

    if (num_dim == 2) //QUad4
    {
        max_nodes_per_patch = 2;
        max_nodes_per_element = 4;
    }

    if (num_dim == 3) //Hex8
    {
        max_nodes_per_patch = 4;
        max_nodes_per_element = 8;
    }

    // copy temporary element storage to multivector storage
    dual_nodes_in_elem = dual_elem_conn_array("dual_nodes_in_elem", rnum_elem, max_nodes_per_element);
    host_elem_conn_array nodes_in_elem = dual_nodes_in_elem.view_host();
    dual_nodes_in_elem.modify_host();

    for (int ielem = 0; ielem < rnum_elem; ielem++)
    {
        for (int inode = 0; inode < elem_words_per_line; inode++)
        {
            nodes_in_elem(ielem, inode) = element_temp[ielem * elem_words_per_line + inode];
        }
    }

    // view storage for all local elements connected to local nodes on this rank
    // DCArrayKokkos<GO, array_layout, device_type, memory_traits> All_Element_Global_Indices(rnum_elem);
    Kokkos::DualView<GO*, array_layout, device_type, memory_traits> All_Element_Global_Indices("All_Element_Global_Indices", rnum_elem);
    // copy temporary global indices storage to view storage
    for (int ielem = 0; ielem < rnum_elem; ielem++)
    {
        All_Element_Global_Indices.h_view(ielem) = global_indices_temp[ielem];
        if (global_indices_temp[ielem] < 0)
        {
            negative_index_found = 1;
        }
    }

    MPI_Allreduce(&negative_index_found, &global_negative_index_found, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (global_negative_index_found)
    {
        if (process_rank == 0)
        {
            std::cout << "Node index less than or equal to zero detected; set \"zero_index_base = true\" " << std::endl;
        }
        exit_solver(0);
    }

    // delete temporary element connectivity and index storage
    std::vector<size_t>().swap(element_temp);
    std::vector<size_t>().swap(global_indices_temp);

    All_Element_Global_Indices.modify_host();
    All_Element_Global_Indices.sync_device();

    // debug print
    /*
    Kokkos::View <GO*, array_layout, device_type, memory_traits> All_Element_Global_Indices_pass("All_Element_Global_Indices_pass",rnum_elem);
    deep_copy(All_Element_Global_Indices_pass, All_Element_Global_Indices.h_view);
    std::cout << " ------------ELEMENT GLOBAL INDICES ON TASK " << process_rank << " --------------"<<std::endl;
    for (int ielem = 0; ielem < rnum_elem; ielem++){
      std::cout << "elem: " << All_Element_Global_Indices_pass(ielem) + 1;
      std::cout << std::endl;
    }
    */

    // construct overlapping element map (since different ranks can own the same elements due to the local node map)
    all_element_map = Teuchos::rcp(new Tpetra::Map<LO, GO, node_type>(Teuchos::OrdinalTraits<GO>::invalid(), All_Element_Global_Indices.d_view, 0, comm));

    // Close mesh input file
    if (process_rank == 0)
    {
        in->close();
    }

} // end read_mesh
