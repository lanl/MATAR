#ifndef MESH_IO_H
#define MESH_IO_H

#include "matar.h"
#include "mesh.h"
#include "state.h"

using namespace mtr;

#include <map>
#include <memory>
#include <cstring>
#include <sys/stat.h>
#include <iostream>
#include <regex>    // for string pattern recoginition
#include <fstream>
#include <sstream>
#include <vector>
#include <string>   
#include <mpi.h>





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
/// \fn PointIndexFromIJK
///
/// \brief Given (i,j,k) coordinates within the Lagrange hex, return an 
/// offset into the local connectivity (PointIds) array. The order parameter
/// must point to an array of 3 integers specifying the order along each 
/// axis of the hexahedron.
///
/////////////////////////////////////////////////////////////////////////////
inline int PointIndexFromIJK(int i, int j, int k, const int* order)
{
    bool ibdy = (i == 0 || i == order[0]);
    bool jbdy = (j == 0 || j == order[1]);
    bool kbdy = (k == 0 || k == order[2]);
    // How many boundaries do we lie on at once?
    int nbdy = (ibdy ? 1 : 0) + (jbdy ? 1 : 0) + (kbdy ? 1 : 0);

    if (nbdy == 3) { // Vertex DOF
        // ijk is a corner node. Return the proper index (somewhere in [0,7]):
        return (i ? (j ? 2 : 1) : (j ? 3 : 0)) + (k ? 4 : 0);
    }

    int offset = 8;
    if (nbdy == 2) { // Edge DOF
        if (!ibdy) { // On i axis
            return (i - 1) + (j ? order[0] - 1 + order[1] - 1 : 0) + (k ? 2 * (order[0] - 1 + order[1] - 1) : 0) + offset;
        }
        if (!jbdy) { // On j axis
            return (j - 1) + (i ? order[0] - 1 : 2 * (order[0] - 1) + order[1] - 1) + (k ? 2 * (order[0] - 1 + order[1] - 1) : 0) + offset;
        }
        // !kbdy, On k axis
        offset += 4 * (order[0] - 1) + 4 * (order[1] - 1);
        return (k - 1) + (order[2] - 1) * (i ? (j ? 3 : 1) : (j ? 2 : 0)) + offset;
    }

    offset += 4 * (order[0] - 1 + order[1] - 1 + order[2] - 1);
    if (nbdy == 1) { // Face DOF
        if (ibdy) { // On i-normal face
            return (j - 1) + ((order[1] - 1) * (k - 1)) + (i ? (order[1] - 1) * (order[2] - 1) : 0) + offset;
        }
        offset += 2 * (order[1] - 1) * (order[2] - 1);
        if (jbdy) { // On j-normal face
            return (i - 1) + ((order[0] - 1) * (k - 1)) + (j ? (order[2] - 1) * (order[0] - 1) : 0) + offset;
        }
        offset += 2 * (order[2] - 1) * (order[0] - 1);
        // kbdy, On k-normal face
        return (i - 1) + ((order[0] - 1) * (j - 1)) + (k ? (order[0] - 1) * (order[1] - 1) : 0) + offset;
    }

    // nbdy == 0: Body DOF
    offset += 2 * ( (order[1] - 1) * (order[2] - 1) + (order[2] - 1) * (order[0] - 1) + (order[0] - 1) * (order[1] - 1));
    return offset + (i - 1) + (order[0] - 1) * ( (j - 1) + (order[1] - 1) * ( (k - 1)));
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



/////////////////////////////////////////////////////////////////////////////
///
/// \fn write_vtk
///
/// \brief Writes a vtk output file
///
/// \param mesh mesh
/// \param node node data
/// \param rank rank
///
/////////////////////////////////////////////////////////////////////////////
    void write_vtk(Mesh_t& mesh,
        node_t& node,
        int rank)
    {

        CArray<double> graphics_times(1);
        int graphics_id = 0;
        graphics_times(0) = 0.0;

        // ---- Update host data ----

        // material point values
        // State.MaterialPoints.den.update_host();
        // State.MaterialPoints.pres.update_host();
        // State.MaterialPoints.stress.update_host();
        // State.MaterialPoints.sspd.update_host();
        // State.MaterialPoints.sie.update_host();
        // State.MaterialPoints.mass.update_host();
        // State.MaterialPoints.conductivity.update_host();
        // State.MaterialPoints.temp_grad.update_host();
        // State.MaterialPoints.eroded.update_host();


        // gauss point values
        // State.GaussPoints.vol.update_host();

        // nodal values
        node.coords.update_host();
        // State.node.vel.update_host();
        // State.node.mass.update_host();
        // State.node.temp.update_host();

        Kokkos::fence();


        const int num_cell_scalar_vars = 2;
        const int num_cell_vec_vars    = 0;
        const int num_cell_tensor_vars = 0;

        const int num_point_scalar_vars = 1;
        const int num_point_vec_vars = 1;


        // Scalar values associated with a cell
        const char cell_scalar_var_names[num_cell_scalar_vars][30] = {
            "rank_id", "elems_in_elem_owned"
        };
        
        // const char cell_vec_var_names[num_cell_vec_vars][15] = {
            
        // };

        const char point_scalar_var_names[num_point_scalar_vars][15] = {
            "rank_id"
        };

        const char point_vec_var_names[num_point_vec_vars][15] = {
            "pos"
        };

        // short hand
        const size_t num_nodes = mesh.num_nodes;
        const size_t num_elems = mesh.num_elems;
        const size_t num_dims  = mesh.num_dims;

        // save the cell state to an array for exporting to graphics files
        auto elem_fields = CArray<double>(num_elems, num_cell_scalar_vars);
        int  elem_switch = 1;


        // save the output scale fields to a single 2D array


        // export material centeric data to the elements
        elem_fields(0, 0) = rank;

        for (size_t elem_gid = 0; elem_gid < mesh.num_elems; elem_gid++) {
            elem_fields(elem_gid, 0) = rank;
            elem_fields(elem_gid, 1) = (double)mesh.num_elems_in_elem(elem_gid);
        }


        // save the vertex vector fields to an array for exporting to graphics files
        CArray<double> vec_fields(num_nodes, num_point_vec_vars, 3);
        CArray<double> point_scalar_fields(num_nodes, num_point_scalar_vars);

        for (size_t node_gid = 0; node_gid < num_nodes; node_gid++) {
            // position, var 0
            vec_fields(node_gid, 0, 0) = node.coords.host(node_gid, 0);
            vec_fields(node_gid, 0, 1) = node.coords.host(node_gid, 1);
            vec_fields(node_gid, 0, 2) = node.coords.host(node_gid, 2);

            point_scalar_fields(node_gid, 0) = rank;
        } // end for loop over vertices


        FILE* out[20];   // the output files that are written to
        char  filename[100]; // char string
        int   max_len = sizeof filename;
        int   str_output_len;

        struct stat st;

        if (stat("vtk", &st) != 0) {
            system("mkdir vtk");
        }

        // snprintf(filename, max_len, "ensight/data/%s.%05d.%s", name, graphics_id, vec_var_names[var]);

        //sprintf(filename, "vtk/Fierro.%05d.vtk", graphics_id);  // mesh file
        str_output_len = snprintf(filename, max_len, "vtk/Fierro.%05d_rank%d.vtk", graphics_id, rank);
        if (str_output_len >= max_len) { fputs("Filename length exceeded; string truncated", stderr); }
         // mesh file
        
        out[0] = fopen(filename, "w");

        fprintf(out[0], "# vtk DataFile Version 2.0\n");  // part 2
        fprintf(out[0], "Mesh for Fierro\n");             // part 2
        fprintf(out[0], "ASCII \n");                      // part 3
        fprintf(out[0], "DATASET UNSTRUCTURED_GRID\n\n"); // part 4

        fprintf(out[0], "POINTS %zu float\n", mesh.num_nodes);

        // write all components of the point coordinates
        for (size_t node_gid = 0; node_gid < mesh.num_nodes; node_gid++) {
            fprintf(out[0],
                    "%f %f %f\n",
                    node.coords.host(node_gid, 0),
                    node.coords.host(node_gid, 1),
                    node.coords.host(node_gid, 2));
        } // end for

        /*
        ---------------------------------------------------------------------------
        Write the elems
        ---------------------------------------------------------------------------
        */

        fprintf(out[0], "\n");
        fprintf(out[0], "CELLS %lu %lu\n", mesh.num_elems, mesh.num_elems + mesh.num_elems * mesh.num_nodes_in_elem);  // size=all printed values

        int Pn_order   = mesh.Pn;
        int order[3]   = { Pn_order, Pn_order, Pn_order };

        // const int num_1D_points = Pn_order+1;

        // write all global point numbers for this elem
        for (size_t elem_gid = 0; elem_gid < mesh.num_elems; elem_gid++) {
            fprintf(out[0], "%lu ", mesh.num_nodes_in_elem); // num points in this elem

            for (int k = 0; k <= Pn_order; k++) {
                for (int j = 0; j <= Pn_order; j++) {
                    for (int i = 0; i <= Pn_order; i++) {
                        size_t node_lid = PointIndexFromIJK(i, j, k, order);
                        fprintf(out[0], "%lu ", mesh.nodes_in_elem.host(elem_gid, node_lid));
                    }
                }
            }

            fprintf(out[0], "\n");
        } // end for

        // Write the element types
        fprintf(out[0], "\n");
        fprintf(out[0], "CELL_TYPES %zu \n", mesh.num_elems);
        // VTK_LAGRANGE_HEXAHEDRON: 72,
        // VTK_HIGHER_ORDER_HEXAHEDRON: 67
        // VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33
        // element types: https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
        // element types: https://kitware.github.io/vtk-js/api/Common_DataModel_CellTypes.html
        // vtk format: https://www.kitware.com//modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
        for (size_t elem_gid = 0; elem_gid < mesh.num_elems; elem_gid++) {
            fprintf(out[0], "%d \n", 72);
        }

        /*
        ---------------------------------------------------------------------------
        Write the nodal vector variables to file
        ---------------------------------------------------------------------------
        */

        fprintf(out[0], "\n");
        fprintf(out[0], "POINT_DATA %zu \n", mesh.num_nodes);

        // vtk vector vars = (position, velocity)
        for (int var = 0; var < num_point_vec_vars; var++) {
            fprintf(out[0], "VECTORS %s float \n", point_vec_var_names[var]);
            for (size_t node_gid = 0; node_gid < mesh.num_nodes; node_gid++) {
                fprintf(out[0], "%f %f %f\n",
                        vec_fields(node_gid, var, 0),
                        vec_fields(node_gid, var, 1),
                        vec_fields(node_gid, var, 2));
            } // end for nodes
        } // end for vec_vars


        // vtk scalar vars = (temp)
        for (int var = 0; var < num_point_scalar_vars; var++) {
            fprintf(out[0], "SCALARS %s float 1\n", point_scalar_var_names[var]);
            fprintf(out[0], "LOOKUP_TABLE default\n");
            for (size_t node_gid = 0; node_gid < mesh.num_nodes; node_gid++) {
                fprintf(out[0], "%f\n",
                        point_scalar_fields(node_gid, 0));
            } // end for nodes
        } // end for vec_vars

        /*
        ---------------------------------------------------------------------------
        Write the scalar elem variable to file
        ---------------------------------------------------------------------------
        */
        fprintf(out[0], "\n");
        fprintf(out[0], "CELL_DATA %zu \n", mesh.num_elems);

        for (int var = 0; var < num_cell_scalar_vars; var++) {
            fprintf(out[0], "SCALARS %s float 1\n", cell_scalar_var_names[var]); // the 1 is number of scalar components [1:4]
            fprintf(out[0], "LOOKUP_TABLE default\n");
            for (size_t elem_gid = 0; elem_gid < mesh.num_elems; elem_gid++) {
                fprintf(out[0], "%f\n",  elem_fields(elem_gid, var));
            } // end for elem
        } // end for cell scalar_vars

        fclose(out[0]);

        // graphics_times(graphics_id) = time_value;

        // Write time series metadata
        //sprintf(filename, "vtk/Fierro.vtk.series", graphics_id);  // mesh file
        str_output_len = snprintf(filename, max_len, "vtk/Fierro.vtk.series"); 
        if (str_output_len >= max_len) { fputs("Filename length exceeded; string truncated", stderr); }
        // mesh file

        out[0] = fopen(filename, "w");

        fprintf(out[0], "{\n");
        fprintf(out[0], "  \"file-series-version\" : \"1.0\",\n");
        fprintf(out[0], "  \"files\" : [\n");

        for (int i = 0; i <= graphics_id; i++) {
            fprintf(out[0], "    { \"name\" : \"Fierro.%05d.vtk\", \"time\" : %12.5e },\n", i, graphics_times(i) );
        }

        // fprintf(out[0], "%12.5e\n", graphics_times(i));
        fprintf(out[0], "  ]\n"); // part 4
        fprintf(out[0], "}"); // part 4

        fclose(out[0]);

        // increment graphics id counter
        // graphics_id++;


    } // end write vtk old



#endif