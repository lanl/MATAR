#ifndef MESH_IO_H
#define MESH_IO_H

#include "matar.h"
#include "mesh.h"
#include "state.h"

using namespace mtr;







/////////////////////////////////////////////////////////////////////////////
///
/// \fn get_id
///
/// \brief This gives the index value of the point or the elem
///
/// Assumes that the grid has an i,j,k structure
/// the elem = i + (j)*(num_points_i-1) + (k)*(num_points_i-1)*(num_points_j-1)
/// the point = i + (j)*num_points_i + (k)*num_points_i*num_points_j
///
/// \param i index
/// \param j index
/// \param k index
/// \param Number of i indices
/// \param Number of j indices
///
/////////////////////////////////////////////////////////////////////////////
inline int get_id(int i, int j, int k, int num_i, int num_j)
{
    return i + j * num_i + k * num_i * num_j;
}

/////////////////////////////////////////////////////////////////////////////
///
/// \fn build_3d_box
///
/// \brief Builds an unstructured 3D rectilinear mesh
///
/// \param Simulation mesh that is built
/// \param Element state data
/// \param Node state data
/// \param origin The origin of the mesh
/// \param length The length of the mesh
/// \param num_elems The number of elements in the mesh
///
/////////////////////////////////////////////////////////////////////////////
void build_3d_box(
    Mesh_t& mesh,
    GaussPoint_t& GaussPoints,
    node_t&   node,
    double origin[3],
    double length[3],
    int num_elems_dim[3])
{
    printf("Creating a 3D box mesh \n");

    const int num_dim = 3;

    // Note: In fierro, these come from the simulation parameters
    const double lx = length[0];
    const double ly = length[1];
    const double lz = length[2];

    // Note: In fierro, these come from the simulation parameters
    const int num_elems_i = num_elems_dim[0];
    const int num_elems_j = num_elems_dim[1];
    const int num_elems_k = num_elems_dim[2];

    const int num_points_i = num_elems_i + 1; // num points in x
    const int num_points_j = num_elems_j + 1; // num points in y
    const int num_points_k = num_elems_k + 1; // num points in y

    const int num_nodes = num_points_i * num_points_j * num_points_k;

    const double dx = lx / ((double)num_elems_i);  // len/(num_elems_i)
    const double dy = ly / ((double)num_elems_j);  // len/(num_elems_j)
    const double dz = lz / ((double)num_elems_k);  // len/(num_elems_k)

    const int num_elems = num_elems_i * num_elems_j * num_elems_k;

    // --- 3D parameters ---
    // const int num_faces_in_elem  = 6;  // number of faces in elem
    // const int num_points_in_elem = 8;  // number of points in elem
    // const int num_points_in_face = 4;  // number of points in a face
    // const int num_edges_in_elem  = 12; // number of edges in a elem

    // initialize mesh node variables
    mesh.initialize_nodes(num_nodes);

        // initialize node state variables, for now, we just need coordinates, the rest will be initialize by the respective solvers
    std::vector<node_state> required_node_state = { node_state::coords };
    node.initialize(num_nodes, num_dim, required_node_state);

    // --- Build nodes ---

    // populate the point data structures
    for (int k = 0; k < num_points_k; k++) {
        for (int j = 0; j < num_points_j; j++) {
            for (int i = 0; i < num_points_i; i++) {
                // global id for the point
                int node_gid = get_id(i, j, k, num_points_i, num_points_j);

                // store the point coordinates
                node.coords.host(node_gid, 0) = origin[0] + (double)i * dx;
                node.coords.host(node_gid, 1) = origin[1] + (double)j * dy;
                node.coords.host(node_gid, 2) = origin[2] + (double)k * dz;
            } // end for i
        } // end for j
    } // end for k


    node.coords.update_device();

    // initialize elem variables
    mesh.initialize_elems(num_elems, num_dim);

    // --- Build elems  ---

    // populate the elem center data structures
    for (int k = 0; k < num_elems_k; k++) {
        for (int j = 0; j < num_elems_j; j++) {
            for (int i = 0; i < num_elems_i; i++) {
                // global id for the elem
                int elem_gid = get_id(i, j, k, num_elems_i, num_elems_j);

                // store the point IDs for this elem where the range is
                // (i:i+1, j:j+1, k:k+1) for a linear hexahedron
                int this_point = 0;
                for (int kcount = k; kcount <= k + 1; kcount++) {
                    for (int jcount = j; jcount <= j + 1; jcount++) {
                        for (int icount = i; icount <= i + 1; icount++) {
                            // global id for the points
                            int node_gid = get_id(icount, jcount, kcount,
                                                num_points_i, num_points_j);

                            // convert this_point index to the FE index convention
                            int this_index = this_point; //convert_point_number_in_Hex(this_point);

                            // store the points in this elem according the the finite
                            // element numbering convention
                            mesh.nodes_in_elem.host(elem_gid, this_index) = node_gid;

                            // increment the point counting index
                            this_point = this_point + 1;
                        } // end for icount
                    } // end for jcount
                }  // end for kcount
            } // end for i
        } // end for j
    } // end for k

    // update device side
    mesh.nodes_in_elem.update_device();



    // Build connectivity
    mesh.build_connectivity();
} // end build_3d_box

#endif