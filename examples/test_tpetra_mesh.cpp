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
#include <stdlib.h>
#include <math.h>  // fmin, fmax, abs note: fminl is long
#include <set>

#include "matar.h"

using namespace mtr; // matar namespace

struct mesh_data {
    int num_dim = 3;
    size_t nlocal_nodes, rnum_elem;
    size_t num_nodes, num_elem;
    TpetraDFArray<double> node_coords_distributed; //unique local coords
    TpetraDFArray<double> ghost_node_coords_distributed; //local data set by other processes
    TpetraDFArray<double> all_node_coords_distributed; //unique + ghost
    TpetraDFArray<long long int> nodes_in_elem_distributed;
};

void setup_maps(mesh_data &mesh);
void read_mesh_vtk(const char* MESH,mesh_data &mesh);
void run_test(mesh_data &mesh);

int main(int argc, char* argv[])
{   
    MPI_Init(&argc, &argv);
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    Kokkos::initialize();
    {
        //allocate mesh struct
        mesh_data mesh;

        // read mesh file
        read_mesh_vtk(argv[1], mesh);

        //setup ghost maps
        setup_maps(mesh);

        //compute something; barriers for timer
        MPI_Barrier(MPI_COMM_WORLD);
        Kokkos::fence();
        
        auto time_1 = std::chrono::high_resolution_clock::now();
        run_test(mesh);
        auto time_2 = std::chrono::high_resolution_clock::now();

        Kokkos::fence();
        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);
        std::chrono::duration <float, std::milli> ms = time_2 - time_1;
        if(process_rank==0)
            std::cout << "Finished. Runtime was " << ms.count() << std::endl;
    } // end of kokkos scope
    Kokkos::finalize();
    MPI_Finalize();
}

/* ----------------------------------------------------------------------
   Construct maps containing ghost nodes
------------------------------------------------------------------------- */
void run_test(mesh_data &mesh)
{   
    int  num_dim = mesh.num_dim;
    TpetraDFArray<double> all_node_coords_distributed = mesh.all_node_coords_distributed;
    TpetraDFArray<long long int> nodes_in_elem_distributed = mesh.nodes_in_elem_distributed;
    TpetraPartitionMap<> all_node_map = mesh.nodes_in_elem_distributed.pmap;
    int ntimesteps = 1000;
    real_t constant_velocity = 0.0001;
    real_t timestep = 0.001;

    //arbitrary calculation done by looping over all local elements for all timesteps
    //this loops over all ghosts as well to test load balancing
    for(int itimestep = 0; itimestep < ntimesteps; itimestep++){
        FOR_ALL(ielem,0,mesh.rnum_elem, {
                for(int inode=0; inode < 8; inode++){
                    int local_node_index = nodes_in_elem_distributed(ielem,inode);
                    for(int idim=0; idim < num_dim; idim++){
                        all_node_coords_distributed(local_node_index, idim) += constant_velocity*timestep;
                    }
                }
        });
    }
}

/* ----------------------------------------------------------------------
   Construct maps containing ghost nodes
------------------------------------------------------------------------- */
void setup_maps(mesh_data &mesh)
{
    int  num_dim = mesh.num_dim;
    int  local_node_index, current_column_index;
    int  nodes_per_element;
    long long int   node_gid;
    TpetraDFArray<double> node_coords_distributed = mesh.node_coords_distributed;
    TpetraDFArray<long long int> nodes_in_elem_distributed = mesh.nodes_in_elem_distributed;
    size_t rnum_elem = mesh.rnum_elem;
    size_t nlocal_nodes = mesh.nlocal_nodes;
    size_t num_nodes = mesh.num_nodes;
    size_t num_elem = mesh.num_elem;
    size_t nghost_nodes, nall_nodes;
    
    TpetraPartitionMap<> map = mesh.node_coords_distributed.pmap;
    DCArrayKokkos<long long int, Kokkos::LayoutLeft> ghost_nodes;
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if (rnum_elem >= 1)
    {
        // Construct set of ghost nodes; start with a buffer with upper limit
        size_t buffer_limit = 0;
        if (num_dim == 2)
        {
            for (int ielem = 0; ielem < rnum_elem; ielem++)
            {
                buffer_limit += 4;
            }
        }

        if (num_dim == 3)
        {
            for (int ielem = 0; ielem < rnum_elem; ielem++)
            {
                buffer_limit += 8;
            }
        }

        CArrayKokkos<size_t, Kokkos::LayoutLeft, HostSpace> ghost_node_buffer(buffer_limit);

        std::set<long long int> ghost_node_set;

        // search through local elements for global node indices not owned by this MPI rank
        if (num_dim == 2)
        {
            for (int cell_rid = 0; cell_rid < rnum_elem; cell_rid++)
            {
                // set nodes per element
                nodes_per_element = 4;
                for (int node_lid = 0; node_lid < nodes_per_element; node_lid++)
                {
                    node_gid = nodes_in_elem_distributed.host(cell_rid, node_lid);
                    if (!map.isProcessGlobalIndex(node_gid))
                    {
                        ghost_node_set.insert(node_gid);
                    }
                }
            }
        }

        if (num_dim == 3)
        {
            for (int cell_rid = 0; cell_rid < rnum_elem; cell_rid++)
            {
                // set nodes per element
                nodes_per_element = 8;
                for (int node_lid = 0; node_lid < nodes_per_element; node_lid++)
                {
                    node_gid = nodes_in_elem_distributed.host(cell_rid, node_lid);
                    if (!map.isProcessGlobalIndex(node_gid))
                    {
                        ghost_node_set.insert(node_gid);
                    }
                }
            }
        }

        // by now the set contains, with no repeats, all the global node indices that are ghosts for this rank
        // now pass the contents of the set over to a CArrayKokkos, then create a map to find local ghost indices from global ghost indices

        nghost_nodes     = ghost_node_set.size();
        ghost_nodes = DCArrayKokkos<long long int, Kokkos::LayoutLeft>(nghost_nodes, "ghost_nodes");
        int  ighost = 0;
        auto it     = ghost_node_set.begin();

        while (it != ghost_node_set.end()) {
            ghost_nodes.host(ighost++) = *it;
            it++;
        }

        // debug print of ghost nodes
        // std::cout << " GHOST NODE SET ON TASK " << process_rank << std::endl;
        // for(int i = 0; i < nghost_nodes; i++)
        // std::cout << "{" << i + 1 << "," << ghost_nodes(i) + 1 << "}" << std::endl;

        // debug print of ghost nodes
        // std::cout << " GHOST NODE MAP ON TASK " << process_rank << std::endl;
        // for(int i = 0; i < nghost_nodes; i++)
        // std::cout << "{" << i + 1 << "," << global2local_map.get(ghost_nodes(i)) + 1 << "}" << std::endl;
    }

    ghost_nodes.update_host();

    // create a Map for ghost node indices
    TpetraPartitionMap<> ghost_node_map = TpetraPartitionMap<>(ghost_nodes);

    // communicate ghost node positions; construct multivector distributed object using local node data

    // construct array for all indices (ghost + local)
    nall_nodes = nlocal_nodes + nghost_nodes;
    // CArrayKokkos<GO, array_layout, device_type, memory_traits> all_node_indices(nall_nodes, "all_node_indices");
    DCArrayKokkos<long long int, Kokkos::LayoutLeft> all_node_indices(nall_nodes, "all_node_indices");
    //map.print();
    for (int i = 0; i < nall_nodes; i++)
    {
        if (i < nlocal_nodes)
        {
            all_node_indices.host(i) = map.getGlobalIndex(i);
        }
        else
        {
            all_node_indices.host(i) = ghost_nodes.host(i - nlocal_nodes);
        }
        //if(all_node_indices.host(i) < 0)
        //std::cout << "NEGATIVE INDEX AT " << i << " WITH LOCAL NODE COUNT " << nlocal_nodes << " " << all_node_indices.host(i) << " " << std::endl;
    }
    all_node_indices.update_device();
    // debug print of node indices
    // for(int inode=0; inode < index_counter; inode++)
    // std::cout << " my_reduced_global_indices " << my_reduced_global_indices(inode) <<std::endl;

    // create a Map for all the node indices (ghost + local)
    TpetraPartitionMap<> all_node_map = TpetraPartitionMap<>(all_node_indices);

    // create distributed multivector of the (local + ghost) node coords
    mesh.all_node_coords_distributed = TpetraDFArray<double>(all_node_map, num_dim);

    // create distributed multivector of the ghost node coords as a subview of the all vector
    mesh.ghost_node_coords_distributed = TpetraDFArray<double>(mesh.all_node_coords_distributed, ghost_node_map, nlocal_nodes);

    // create communication object between ghosts and unique local data
    TpetraCommunicationPlan<real_t> ghost_comms(mesh.ghost_node_coords_distributed, mesh.node_coords_distributed);

    // comms to get ghosts coords initialized
    ghost_comms.execute_comms();

    //initialize 0:nlocal-1 data in the all vector since the comms just set nlocal:nall via the subview
    FOR_ALL(inode,0,nlocal_nodes, {
        for (int idim=0; idim < num_dim; idim++){
            mesh.all_node_coords_distributed(inode,idim) = mesh.node_coords_distributed(inode,idim);
        }
    });
    mesh.all_node_coords_distributed.update_host();

    //convert nodes in elem to local node ids to avoid excessive map conversion calls
    FOR_ALL(ielem,0,mesh.rnum_elem, {
        for(int inode=0; inode < 8; inode++){
            //recall that nodes in elem is storing global indices in this implementation
            //you may just want to store local indices in your case to avoid the map call
            nodes_in_elem_distributed(ielem,inode) = all_node_map.getLocalIndex(nodes_in_elem_distributed(ielem,inode));
        }
    });

    // std::cout << "number of patches = " << mesh->num_patches() << std::endl;
    if (process_rank == 0)
    {
        std::cout << "End of map setup " << std::endl;
    }
}

/* ----------------------------------------------------------------------
   Read VTK format mesh file
------------------------------------------------------------------------- */

void read_mesh_vtk(const char* MESH, mesh_data &mesh)
{
    std::string skip_line, read_line, substring;
    std::stringstream line_parse;

    int num_dim = mesh.num_dim;
    int local_node_index, current_column_index;
    int buffer_loop, buffer_iteration, buffer_iterations, dof_limit, scan_loop;
    int negative_index_found = 0;
    int global_negative_index_found = 0;

    size_t read_index_start, node_rid, elem_gid;
    size_t strain_count;
    size_t nlocal_nodes, rnum_elem, max_nodes_per_element;
    size_t buffer_nlines = 100000;
    size_t num_nodes, num_elem;
    size_t max_word = 30;

    long long int     node_gid;
    int words_per_line, elem_words_per_line;
    real_t dof_value;
    real_t unit_scaling      = 1.0;
    bool   zero_index_base   = true;

    std::ifstream* in = NULL;
    std::string filename(MESH);

    CArrayKokkos<char, Kokkos::LayoutLeft, HostSpace> read_buffer;

    //corresponding MPI rank for this process
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // read the mesh
    // --- Read the number of nodes in the mesh --- //
    num_nodes = 0;
    if (process_rank == 0)
    {
        std::cout << "FILE NAME IS " << filename << std::endl;
        std::cout << " NUM DIM is " << num_dim << std::endl;
        in = new std::ifstream();
        in->open(filename);
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
    MPI_Bcast(&num_nodes, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    mesh.num_nodes = num_nodes;

    // construct distributed storage for node coordinates now that we know the global number of nodes
    // the default map of this array assigns an ordered contiguous subset of the global IDs to each process
    mesh.node_coords_distributed = TpetraDFArray<double>(num_nodes, num_dim);
    // node_coords_distributed.pmap.print();

    //map of the distributed node coordinates vector
    TpetraPartitionMap<> map = mesh.node_coords_distributed.pmap;

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
    read_buffer = CArrayKokkos<char, Kokkos::LayoutLeft, HostSpace>(buffer_nlines, words_per_line, max_word);

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
        MPI_Bcast(read_buffer.pointer(), buffer_nlines * words_per_line * max_word, MPI_CHAR, 0, MPI_COMM_WORLD);
        // broadcast how many nodes were read into this buffer iteration
        MPI_Bcast(&buffer_loop, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
                mesh.node_coords_distributed.host(node_rid, 0) = dof_value * unit_scaling;
                dof_value = atof(&read_buffer(scan_loop, 1, 0));
                mesh.node_coords_distributed.host(node_rid, 1) = dof_value * unit_scaling;
                if (num_dim == 3)
                {
                    dof_value = atof(&read_buffer(scan_loop, 2, 0));
                    mesh.node_coords_distributed.host(node_rid, 2) = dof_value * unit_scaling;
                }
            }
        }
        read_index_start += buffer_nlines;
    }
    // repartition node distribution
    mesh.node_coords_distributed.update_device();
    mesh.node_coords_distributed.repartition_vector();
    //reset our local map variable to the repartitioned map
    map = mesh.node_coords_distributed.pmap;
    // set the local number of nodes on this process
    mesh.nlocal_nodes = nlocal_nodes = mesh.node_coords_distributed.dims(0);

    // synchronize device data


    // check that local assignments match global total

    // read in element info (ensight file format is organized in element type sections)
    // loop over this later for several element type sections

    num_elem  = 0;
    rnum_elem = 0;
    CArrayKokkos<int, Kokkos::LayoutLeft, HostSpace> node_store(elem_words_per_line);

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
    MPI_Bcast(&num_elem, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    mesh.num_elem = num_elem;

    if (process_rank == 0)
    {
        std::cout << "before mesh initialization" << std::endl;
    }

    // read in element connectivity
    // we're gonna reallocate for the words per line expected for the element connectivity
    read_buffer = CArrayKokkos<char, Kokkos::LayoutLeft, HostSpace>(buffer_nlines, elem_words_per_line, max_word);

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
        MPI_Bcast(read_buffer.pointer(), buffer_nlines * elem_words_per_line * max_word, MPI_CHAR, 0, MPI_COMM_WORLD);
        // broadcast how many nodes were read into this buffer iteration
        MPI_Bcast(&buffer_loop, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
        max_nodes_per_element = 4;
    }

    if (num_dim == 3) //Hex8
    {
        max_nodes_per_element = 8;
    }

    mesh.rnum_elem = rnum_elem;

    // copy temporary element storage to multivector storage
    DCArrayKokkos<long long int> All_Element_Global_Indices(rnum_elem, "dual_nodes_in_elem");

    // copy temporary global indices storage to view storage
    for (int ielem = 0; ielem < rnum_elem; ielem++)
    {
        All_Element_Global_Indices.host(ielem) = global_indices_temp[ielem];
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
    }

    All_Element_Global_Indices.update_device();

    //map object with distribution of global indices of all Elements each process stores
    TpetraPartitionMap<> all_element_map(All_Element_Global_Indices);

    //build nodes in elem distributed storage
    mesh.nodes_in_elem_distributed = TpetraDFArray<long long int>(all_element_map, max_nodes_per_element, "nodes_in_elem_distributed");

    for (int ielem = 0; ielem < rnum_elem; ielem++)
    {
        for (int inode = 0; inode < elem_words_per_line; inode++)
        {
            mesh.nodes_in_elem_distributed.host(ielem, inode) = element_temp[ielem * elem_words_per_line + inode];
        }
    }
    mesh.nodes_in_elem_distributed.update_device();

    // delete temporary element connectivity and index storage
    std::vector<size_t>().swap(element_temp);
    std::vector<size_t>().swap(global_indices_temp);

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
    //nodes_in_elem_distributed.print();

    // Close mesh input file
    if (process_rank == 0)
    {
        in->close();
    }

} // end read_mesh
