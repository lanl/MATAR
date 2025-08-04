#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <cfloat>
#include <iomanip>
#include <algorithm>
#include "matar.h"
#include <Kokkos_Core.hpp>
#include <chrono>   // for timing

// using namespace std; // Enables std::
using namespace mtr; // MATAR

KOKKOS_INLINE_FUNCTION // This function does ray-triangle intersection
bool intersects_with_edge_detect(const double orig[3], const double dir[3], const double v0[3], const double v1[3], const double v2[3], bool& edge_hit) { // extra flag
    const double EPS = 1e-8; // Small epsilon in order to avoid errors with comparisons
    const double EDGE_EPS = 1e-5; // Tolerance for edge detection
    double edge1[3], edge2[3], h[3], s[3], q[3]; // Temporary vectors used for intersection test
    for (int i = 0; i < 3; ++i) { // Computes the edges of the triangle
        edge1[i] = v1[i] - v0[i];
        edge2[i] = v2[i] - v0[i];
    }

    h[0] = dir[1] * edge2[2] - dir[2] * edge2[1]; // Cross product
    h[1] = dir[2] * edge2[0] - dir[0] * edge2[2]; // Cross product
    h[2] = dir[0] * edge2[1] - dir[1] * edge2[0]; // Cross product
    double a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2]; // Dot product
    if (fabs(a) < EPS) return false; // If a is close to zero then then the ray is parallel to the riangle which means there is no intersection
    double f = 1.0 / a; // Precomputes reciprocal
    for (int i = 0; i < 3; ++i) s[i] = orig[i] - v0[i]; // Vector from the triangle vertex to the ray origin
    double u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]); // Computes the barycentric coordinate which is u
    if (u < 0.0 || u > 1.0) return false; // If u is out of range then the intercestion is outsdide of the triangle
    q[0] = s[1] * edge1[2] - s[2] * edge1[1]; // Cross product
    q[1] = s[2] * edge1[0] - s[0] * edge1[2]; // Cross product
    q[2] = s[0] * edge1[1] - s[1] * edge1[0]; // Cross product
    double v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]); // Computes the barycentric coordinate which is v
    if (v < 0.0 || u + v > 1.0) return false; // Checks if inside of the triangle
    double t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]); // Computes the distance t along the ray to the intersection point
    if (t <= EPS) return false;
    edge_hit = (u < EDGE_EPS || u > 1.0 - EDGE_EPS || v < EDGE_EPS || v > 1.0 - EDGE_EPS || u + v > 1.0 - EDGE_EPS);
    return true;
}

// Newton-Raphson Inversion
KOKKOS_INLINE_FUNCTION
bool invert_mapping_newton(
    const double x_target[3],
    double x_nodes_1D[24],
    double xi_eta_zeta[3],
    const int max_iter = 30,
    const double tol = 1e-12) {

    // inside function
    double xi = 0.0, eta = 0.0, zeta = 0.0; // Initial guess

    ViewCArrayDevice <double> x_nodes(&x_nodes_1D[0], 8, 3);

    double F[3]; // The residual vector
    double J_1D[9]; // Jacobian Matrix
    double invJ_1D[9]; // Declares the inverse matrix of the Jacobian matrix
    ViewCArrayDevice <double> J(&J_1D[0], 3, 3);
    ViewCArrayDevice <double> invJ(&invJ_1D[0], 3, 3);

    // WARNING: replace lambda function
    auto phi = [](int p, double xi, double eta, double zeta) { // Defines the shape function
        int sx = (p & 1) ? +1 : -1; // Sign of the xi derivative
        int sy = (p & 2) ? +1 : -1; // Sign of the eta derivative
        int sz = (p & 4) ? +1 : -1; // Sign of the zeta derivative
        return 0.125 * (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta);
        };

    // WARNING: replace lambda function
    auto dphi = [](int p, double xi, double eta, double zeta, double* out) {
        int sx = (p & 1) ? +1 : -1;
        int sy = (p & 2) ? +1 : -1;
        int sz = (p & 4) ? +1 : -1;

        out[0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta);  // dphi/dxi
        out[1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta);  // dphi/deta
        out[2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta);   // dphi/dzeta
        };

    // iterate
    for (size_t it = 0; it < max_iter; ++it) { // Iterative Newton method

        // initiale F (residual vector) and J to 0
        for (size_t i = 0; i < 3; i++) {
            F[i] = 0.0;
            for (size_t j = 0; j < 3; j++) {
                J(i, j) = 0.0;
            } // end for j
        } // end for i

        // loop over nodes
        for (int p = 0; p < 8; ++p) { // loop over all 8 nodes of the hexahedral element

            double phi_p = phi(p, xi, eta, zeta);
            double dphi_vals[3];

            dphi(p, xi, eta, zeta, dphi_vals);

            for (int d = 0; d < 3; ++d) {
                F[d] += phi_p * x_nodes(p, d);
                J(d, 0) += dphi_vals[0] * x_nodes(p, d);  // dphi/dxi
                J(d, 1) += dphi_vals[1] * x_nodes(p, d);  // dphi/deta
                J(d, 2) += dphi_vals[2] * x_nodes(p, d);  // dphi/dzeta
            }
        } // end for over nodes

        for (int d = 0; d < 3; ++d) F[d] -= x_target[d]; // Compute the residual

        double nrm = sqrt(F[0] * F[0] + F[1] * F[1] + F[2] * F[2]); // The L2 norm
        if (nrm < tol) { // Checks convergence, if the residual is smaller than the tolerance there is a solution
            xi_eta_zeta[0] = xi; // Stores converged xi coordinate
            xi_eta_zeta[1] = eta; // Stores converged eta coordinate
            xi_eta_zeta[2] = zeta; // Stores converged zeta coordinate
            return true; // It converged
        }

        double det = // Compute the inverse Jacoban by the determinant
            J(0, 0) * (J(1, 1) * J(2, 2) - J(1, 2) * J(2, 1)) - // Compute the inverse Jacoban by the determinant
            J(0, 1) * (J(1, 0) * J(2, 2) - J(1, 2) * J(2, 0)) + // Compute the inverse Jacoban by the determinant
            J(0, 2) * (J(1, 0) * J(2, 1) - J(1, 1) * J(2, 0)); // Compute the inverse Jacoban by the determinant

        if (fabs(det) < 1e-16) return false; // Singular Jacobian matrix

        invJ(0, 0) = (J(1, 1) * J(2, 2) - J(1, 2) * J(2, 1)) / det; // Cofactor expansion for entry [0][0] of the inverse matrix
        invJ(0, 1) = (J(0, 2) * J(2, 1) - J(0, 1) * J(2, 2)) / det; // Cofactor expansion for entry [0][1] of the inverse matrix
        invJ(0, 2) = (J(0, 1) * J(1, 2) - J(0, 2) * J(1, 1)) / det; // Cofactor expansion for entry (0,2) of the inverse matrix

        invJ(1, 0) = (J(1, 2) * J(2, 0) - J(1, 0) * J(2, 2)) / det; // Cofactor expansion for entry [1][0] of the inverse matrix
        invJ(1, 1) = (J(0, 0) * J(2, 2) - J(0, 2) * J(2, 0)) / det; // Cofactor expansion for entry [1][1] of the inverse matrix
        invJ(1, 2) = (J(0, 2) * J(1, 0) - J(0, 0) * J(1, 2)) / det; // Cofactor expansion for entry [1][2] of the inverse matrix

        invJ(2, 0) = (J(1, 0) * J(2, 1) - J(1, 1) * J(2, 0)) / det; // Cofactor expansion for entry [2][0] of the inverse matrix
        invJ(2, 1) = (J(0, 1) * J(2, 0) - J(0, 0) * J(2, 1)) / det; // Cofactor expansion for entry [2][1] of the inverse matrix
        invJ(2, 2) = (J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0)) / det; // Cofactor expansion for entry [2][2] of the inverse matrix

        double dxi = invJ(0, 0) * F[0] + invJ(0, 1) * F[1] + invJ(0, 2) * F[2]; // Computes the correction in the xi direction
        double deta = invJ(1, 0) * F[0] + invJ(1, 1) * F[1] + invJ(1, 2) * F[2]; // Computes the correction in the eta direction
        double dzeta = invJ(2, 0) * F[0] + invJ(2, 1) * F[1] + invJ(2, 2) * F[2]; // Computes the correction in the zeta direction

        xi -= dxi; // Update xi using Newtonian step
        eta -= deta; // Update eta using Newtonian step
        zeta -= dzeta; // Update zeta using Newtonian step

    } // end for iterations

    return false; // There was no convergence

} // end function

// -----------------
///>  This function checks if a ray intersected an axis aligned bounding box
///>  partical_landing_coord is the stopping location for the partical thrown
///>  dir is a unit vector in the direction of the partical being thrown
///>  bounds_min is the min for the entire STL cad file
///>  bounds_max is the max for the entire STL cad file
KOKKOS_INLINE_FUNCTION
bool RayIntersectsAxisAlignedBox(const double partical_landing_coord[3],
    const double dir[3],
    const double bounds_min[3],
    const double bounds_max[3]) {

    double tmin = -DBL_MAX; // Starts the minimum intersection distance as -infinity
    double tmax = DBL_MAX; // Starts the maximum intersection distance as infinity

    for (int i = 0; i < 3; ++i) { // Loops over the x, y and z axes

        if (fabs(dir[i]) < 1e-8) { // The the ray is closely parallel to the axis

            // If the partical_landing_coord is outside of the box there is no intersection 
            if (partical_landing_coord[i] < bounds_min[i] || partical_landing_coord[i] > bounds_max[i]) return false;

        }
        else {
            double invD = 1.0 / dir[i]; // Computes the inverse of the direction component
            double t0 = (bounds_min[i] - partical_landing_coord[i]) * invD; // Distance to the close plane
            double t1 = (bounds_max[i] - partical_landing_coord[i]) * invD; // Distance to the far plane
            double t3;

            if (t0 > t1) {
                t3 = t0;
                t0 = t1;
                t1 = t3;
            }
            // std::swap(t0, t1); // Guarantees t0 is close and t1 is far

            tmin = fmax(tmin, t0); // Updates the global entry point
            tmax = fmin(tmax, t1); // Updates the global exit point 
            if (tmax < tmin) return false; // No intersection if the intervals don't overlap
        } // end if
    } // loop over demensions
    return true; // All of the axes overlap
}

KOKKOS_INLINE_FUNCTION
size_t grid_index(size_t ix, size_t iy, size_t iz, size_t Ny, size_t Nz) {
    return iz * Ny * Nz + iy * Ny + ix;
}

std::tuple<
    CArray<float>,   // normal
    CArray<float>, CArray<float>, CArray<float>,   // v1X, v1Y, v1Z
    CArray<float>, CArray<float>, CArray<float>,   // v2X, v2Y, v2Z
    CArray<float>, CArray<float>, CArray<float>,   // v3X, v3Y, v3Z
    unsigned int                           // n_facets
>
binary_stl_reader(const std::string& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { std::perror("open"); std::exit(EXIT_FAILURE); }

    const std::streamoff filesize = in.tellg();
    if (filesize < 100) {
        std::cerr << "ERROR: File too small to be a valid STL\n";
        std::exit(EXIT_FAILURE);
    }
    in.seekg(0);

    // ---- check if ASCII -------------------------------------------------
    char magic[6] = { 0 };
    in.read(magic, 5);          // read first 5 chars
    in.seekg(0);               // rewind
    if (std::strncmp(magic, "solid", 5) == 0) {
        std::cerr
            << "ERROR: \"" << path
            << "\" looks like an **ASCII** STL (starts with \"solid\").\n"
            << "Re‑export it as *binary* or implement an ASCII parser.\n";
        std::exit(EXIT_FAILURE);        // or call ascii_stl_reader();
    }

    // ---- read 80‑byte header + nominal facet count ----------------------
    char header[80];                in.read(header, 80);
    unsigned int n_facets_nominal;  in.read(reinterpret_cast<char*>(&n_facets_nominal), 4);

    // ---- compute expected count from file size to sanity‑check ----------
    // binary facet record = 50 bytes (12×4 + 12×4 + 12×4 + 2)
    const unsigned int n_facets_from_size =
        static_cast<unsigned int>((filesize - 84) / 50);

    unsigned int n_facets = n_facets_nominal;
    if (n_facets_nominal != n_facets_from_size) {
        std::cout << "WARNING: facet count in header (" << n_facets_nominal
            << ") disagrees with file size (" << n_facets_from_size
            << ").  Using size‑derived value.\n";
        n_facets = n_facets_from_size;
    }
    std::cout << "STL facets: " << n_facets << '\n';

    // ---- allocate MATAR arrays -----------------------------------------
    CArray<float> normal(n_facets, 3);
    CArray<float> v1X(n_facets), v1Y(n_facets), v1Z(n_facets);
    CArray<float> v2X(n_facets), v2Y(n_facets), v2Z(n_facets);
    CArray<float> v3X(n_facets), v3Y(n_facets), v3Z(n_facets);

    // ---- read facet records --------------------------------------------
    float nrm[3], v1[3], v2[3], v3[3];
    for (unsigned int i = 0; i < n_facets; ++i) {
        in.read(reinterpret_cast<char*>(nrm), 12);
        in.read(reinterpret_cast<char*>(v1), 12);
        in.read(reinterpret_cast<char*>(v2), 12);
        in.read(reinterpret_cast<char*>(v3), 12);
        in.ignore(2);                        // attribute byte count

        for (int d = 0; d < 3; ++d) normal(i, d) = nrm[d];
        v1X(i) = v1[0]; v1Y(i) = v1[1]; v1Z(i) = v1[2];
        v2X(i) = v2[0]; v2Y(i) = v2[1]; v2Z(i) = v2[2];
        v3X(i) = v3[0]; v3Y(i) = v3[1]; v3Z(i) = v3[2];
    }
    return { normal,v1X,v1Y,v1Z,v2X,v2Y,v2Z,v3X,v3Y,v3Z,n_facets };
}

// ---------------
// main function
// ---------------
int main(int argc, char* argv[]) { // Start of the main function

    double Lx, Ly, Lz; // Dimensions of the bounding box
    size_t Nx, Ny, Nz; // The number of elemss along each axis

    Kokkos::initialize(argc, argv); 
{ // Initializes Kokkos

//      stl_reader::StlMesh<float, unsigned int> mesh("/home/jbenner/MATAR/examples/square_pyramid.stl"); // Reads the STL file

    auto stl_data = binary_stl_reader("/var/tmp/repos/MATAR/Sphere.stl");
    auto& normal = std::get<0>(stl_data);
    auto& v1X = std::get<1>(stl_data);
    auto& v1Y = std::get<2>(stl_data);
    auto& v1Z = std::get<3>(stl_data);
    auto& v2X = std::get<4>(stl_data);
    auto& v2Y = std::get<5>(stl_data);
    auto& v2Z = std::get<6>(stl_data);
    auto& v3X = std::get<7>(stl_data);
    auto& v3Y = std::get<8>(stl_data);
    auto& v3Z = std::get<9>(stl_data);
    auto& n_tris = std::get<10>(stl_data);

    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

    for (size_t i = 0; i < n_tris; ++i) {
        float xs[3] = { v1X(i), v2X(i), v3X(i) };
        float ys[3] = { v1Y(i), v2Y(i), v3Y(i) };
        float zs[3] = { v1Z(i), v2Z(i), v3Z(i) };

        for (int j = 0; j < 3; ++j) {
            min_x = fmin(min_x, xs[j]);
            min_y = fmin(min_y, ys[j]);
            min_z = fmin(min_z, zs[j]);
            max_x = fmax(max_x, xs[j]);
            max_y = fmax(max_y, ys[j]);
            max_z = fmax(max_z, zs[j]);
        }
    }

    const float cx = 0.5f * (min_x + max_x);
    const float cy = 0.5f * (min_y + max_y);
    const float cz = 0.5f * (min_z + max_z);
    const float scale = 2.f / std::max({ max_x - min_x, max_y - min_y, max_z - min_z });

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Bounding Box Dimensions:\n";
    std::cout << "  Dx = " << max_x - min_x << "\n";
    std::cout << "  Dy = " << max_y - min_y << "\n";
    std::cout << "  Dz = " << max_z - min_z << "\n";

    // === Prepare verts_in_tri and tri_min/max ===
    CArrayDual <double> verts_in_tri(n_tris, 3, 3, "verts_in_tri");
    CArrayDual <double> tri_min(n_tris, 3, "tri_min");
    CArrayDual <double> tri_max(n_tris, 3, "tri_max");

    FOR_LOOP(t, 0, n_tris, {
        // verts_in_tri.host(t, 0, 0) = (v1X(t) - cx) * scale;
        // verts_in_tri.host(t, 0, 1) = (v1Y(t) - cy) * scale;
        // verts_in_tri.host(t, 0, 2) = (v1Z(t) - cz) * scale;

        // verts_in_tri.host(t, 1, 0) = (v2X(t) - cx) * scale;
        // verts_in_tri.host(t, 1, 1) = (v2Y(t) - cy) * scale;
        // verts_in_tri.host(t, 1, 2) = (v2Z(t) - cz) * scale;

        // verts_in_tri.host(t, 2, 0) = (v3X(t) - cx) * scale;
        // verts_in_tri.host(t, 2, 1) = (v3Y(t) - cy) * scale;
        // verts_in_tri.host(t, 2, 2) = (v3Z(t) - cz) * scale;

        verts_in_tri.host(t, 0, 0) = (v1X(t) - cx) * scale;
        verts_in_tri.host(t, 0, 1) = (v1Y(t) - cy) * scale;
        verts_in_tri.host(t, 0, 2) = (v1Z(t) - cz) * scale;

        verts_in_tri.host(t, 1, 0) = (v2X(t) - cx) * scale;
        verts_in_tri.host(t, 1, 1) = (v2Y(t) - cy) * scale;
        verts_in_tri.host(t, 1, 2) = (v2Z(t) - cz) * scale;

        verts_in_tri.host(t, 2, 0) = (v3X(t) - cx) * scale;
        verts_in_tri.host(t, 2, 1) = (v3Y(t) - cy) * scale;
        verts_in_tri.host(t, 2, 2) = (v3Z(t) - cz) * scale;

        for (size_t dim = 0; dim < 3; ++dim) {
            tri_min.host(t, dim) = fmin(fmin(verts_in_tri.host(t, 0, dim), verts_in_tri.host(t, 1, dim)), verts_in_tri.host(t, 2, dim));
            tri_max.host(t, dim) = fmax(fmax(verts_in_tri.host(t, 0, dim), verts_in_tri.host(t, 1, dim)), verts_in_tri.host(t, 2, dim));
        }
    });
    Kokkos::fence();
    verts_in_tri.update_device();

    std::cout << "Enter the dimensions of the bounding box  (Lx Ly Lz)  : "; // Prompts the user to enter the dimensions of the bounding box
    std::cin >> Lx >> Ly >> Lz; // Reads the dimensions of the bounding box
    std::cout << "Enter the the number of elems along each axis  (Nx Ny Nz)  : "; // Prompts the user to enter the number of elems along each axis
    std::cin >> Nx >> Ny >> Nz; // Reads the number of elems along each axis

    const double geom_dx = 1.0;
    const double geom_dy = 1.0;
    const double geom_dz = 1.0;

    if (Lx < max_x - min_x || Ly < max_y - min_y || Lz < max_z - min_z) {
        std::cerr << "ERROR: bounding box (" << Lx << ',' << Ly << ',' << Lz
            << ") is smaller than the geometry ("
            << max_x - min_x << ',' << max_y - min_y << ','
            << max_z - min_z << ")\n";
        std::exit(EXIT_FAILURE);
    }

    // make a single mesh here, use it everywhere
    const double hx = Lx / ((double)Nx); // Computes the spacing in x
    const double hy = Ly / ((double)Ny); // Computes the spacing in y
    const double hz = Lz / ((double)Nz); // Computes the spacing in z

    const size_t num_elems = Nx * Ny * Nz;
    const size_t num_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1);
    CArrayDual <double> nodes_in_elem(num_elems, 8, "nodes_in_elem");
    CArrayDual <double> node_coords(num_nodes, 3, "nodes_coords");
    CArrayDual <double> elem_vol_frac(num_elems, "elem_vol_frac");
    CArrayDevice <double> ref_cell_verts_in_tri(num_elems, n_tris, 3, 3, "ref_cell_verts_in_tri");
    
    CArrayDevice <double> ref_cell_tri_max(num_elems, n_tris, 3, "ref_cell_tri_max");
    CArrayDevice <double> ref_cell_tri_min(num_elems, n_tris, 3, "ref_cell_tri_min");

    CArrayDual <double> ref_node_coords(8, 3);

    // loop over elems
    FOR_ALL(i, 0, Nx,
            j, 0, Ny,
            k, 0, Nz, {

        // 1D index for cell
        const size_t elem_gid = grid_index(i,j,k,Ny,Nz);

        size_t counter = 0;
        for (size_t i_node = i; i_node <= i + 1; i_node++) {
            for (size_t j_node = j; j_node <= j + 1; j_node++) {
                for (size_t k_node = k; k_node <= k + 1; k_node++) {

                    const size_t node_gid = grid_index(i_node, j_node, k_node, Ny + 1, Nz + 1);

                    // save the grid index for this node of the element
                    nodes_in_elem(elem_gid, counter) = node_gid;

                    // save the node coords for the node at this grid id
                    // node_coords(node_gid, 0) = ((double)i_node) * hx - 0.5 * Lx;
                    // node_coords(node_gid, 1) = ((double)j_node) * hy - 0.5 * Ly;
                    // node_coords(node_gid, 2) = ((double)k_node) * hz - 0.5 * Lz;

                    node_coords(node_gid, 0) = ((double)i_node) * hx - 0.5 * Lx;
                    node_coords(node_gid, 1) = ((double)j_node) * hy - 0.5 * Ly;
                    node_coords(node_gid, 2) = ((double)k_node) * hz - 0.5 * Lz;

                    counter++;
                } // end for k_node
            } // end for j_node
        } // end for i_node

    }); // end parallel over i,j,k of elems of a mesh

    // reference element nodes
    for (size_t i_node = 0; i_node <= 1; i_node++) {
        for (size_t j_node = 0; j_node <= 1; j_node++) {
            for (size_t k_node = 0; k_node <= 1; k_node++) {

                const size_t node_gid = grid_index(i_node, j_node, k_node, 2, 2);

                // save the node coords for the node at this grid id
                ref_node_coords.host(node_gid, 0) = ((double)i_node) * 2.0 - 1.0;
                ref_node_coords.host(node_gid, 1) = ((double)j_node) * 2.0 - 1.0;
                ref_node_coords.host(node_gid, 2) = ((double)k_node) * 2.0 - 1.0;

            } // end for k_node
        } // end for j_node
    } // end for i_node
    ref_node_coords.update_device();

    // ------------
    // I/O is done
    // ------------

    auto time_1 = std::chrono::high_resolution_clock::now();

    // Newton-Raphson FOR_ALL loop over every cell in mesh
    FOR_ALL(elem_gid, 0, num_elems, {

        double node_coords_in_elem_1D[24];
        ViewCArrayDevice <double> node_coords_in_elem(&node_coords_in_elem_1D[0], 8, 3);
        for (size_t node_lid = 0; node_lid < 8; node_lid++) {

            size_t node_gid = nodes_in_elem(elem_gid, node_lid);
            for (size_t dim = 0; dim < 3; dim++) {
                node_coords_in_elem(node_lid, dim) = node_coords(node_gid, dim);
            }
        } // end for loop over nodes of the elem

        bool ok = false;
        for (size_t tri = 0; tri < n_tris; tri++) {
            for (size_t vert = 0; vert < 3; vert++) {

                // goal, find (xi,eta,zeta) for all the verts in a triangle, in the reference element for this cell
                ok = invert_mapping_newton(&verts_in_tri(tri, vert, 0),
                    &node_coords_in_elem(0, 0),
                    &ref_cell_verts_in_tri(elem_gid, tri, vert, 0));

            } // end loop verts in tri

            for (size_t dim = 0; dim < 3; ++dim) {
                ref_cell_tri_min(elem_gid, tri, dim) =
                    fmin(fmin(ref_cell_verts_in_tri(elem_gid, tri, 0, dim),
                        ref_cell_verts_in_tri(elem_gid, tri, 1, dim)),
                        ref_cell_verts_in_tri(elem_gid, tri, 2, dim));

                ref_cell_tri_max(elem_gid, tri, dim) =
                    fmax(fmax(ref_cell_verts_in_tri(elem_gid, tri, 0, dim),
                        ref_cell_verts_in_tri(elem_gid, tri, 1, dim)),
                        ref_cell_verts_in_tri(elem_gid, tri, 2, dim));
            }

        } // end loop over all tris in STL file

        if (!ok) printf("Newton failed in cell %d\n", elem_gid);

    }); // end parallel for over all elems

    const double dir_raw[3] = { 1.0, 0.371, 0.192 }; // Direction vector for ray casting
    const double nrm1 = sqrt(dir_raw[0] * dir_raw[0] + dir_raw[1] * dir_raw[1] + dir_raw[2] * dir_raw[2]); // Computes the Euclidean norm of the direction vector
    const double DIR[3] = { dir_raw[0] / nrm1, dir_raw[1] / nrm1, dir_raw[2] / nrm1 }; // Normalizes the direction vector

    const double dir_alt[3] = { 0.6, 0.8, 0.3 }; // Alternate direction vector for secondary ray casting
    const double nrm2 = sqrt(dir_alt[0] * dir_alt[0] + dir_alt[1] * dir_alt[1] + dir_alt[2] * dir_alt[2]); // Computes the Euclidean norm of the alternate direction vector
    const double DIR2[3] = { dir_alt[0] / nrm2, dir_alt[1] / nrm2, dir_alt[2] / nrm2 }; // Normalizes the alternate direction vector

    const double m_values[8] = { 1, 2, 4, 8, 16, 32, 64, 128 }; // Maximum resolution of particles in elem per axis direction

    for (size_t m_id = 0; m_id < 8; m_id++) { // Loops over increasing elems grid sizes and initializes counters

        const double m = m_values[m_id];

        FOR_ALL(elem_gid, 0, num_elems, {

            bool corner_in[8]; // An array that stores whether the 8 corners of the elems are inside the stl

            for (int c = 0; c < 8; ++c) { // Loops over all 8 corners of the elems

                // this is the point of the reference vertex
                double landing_pt[3];
                landing_pt[0] = ref_node_coords(c,0);
                landing_pt[1] = ref_node_coords(c,1);
                landing_pt[2] = ref_node_coords(c,2);

                int hits = 0; // Counts how many triangles that a ray from the point intersects

                // Loops over all of the triangles in the stl
                for (size_t tri = 0; tri < n_tris; ++tri) {

                    // see if particle landing at vertex of reference element is inside CAD part
                    if (!RayIntersectsAxisAlignedBox(landing_pt,
                                                DIR,
                                                &ref_cell_tri_min(elem_gid, tri, 0),
                                                &ref_cell_tri_max(elem_gid, tri, 0))) {
                        continue;
                    } // Skips the triangle if the ray doesn't intersect the bounding box 

                    bool dummy = false; // Dummy variable
                    if (intersects_with_edge_detect(landing_pt,
                                                    DIR,
                                                    &ref_cell_verts_in_tri(elem_gid, tri, 0, 0),
                                                    &ref_cell_verts_in_tri(elem_gid, tri, 1, 0),
                                                    &ref_cell_verts_in_tri(elem_gid, tri, 2, 0),
                                                    dummy)) {
                            ++hits;
                    } // If a ray hits increment the count

                } // end loop over all triangles in STL file

                corner_in[c] = (hits & 1); // Uses the even-odd rule to tell if the corner is inside the mesh if it hits an odd number

            } // end loop over corners

            bool all_in = true; // Flags in order to determine if the elems is completely inside
            bool all_out = true; // Flags in order to determine if the elems is completely outside

            for (int c = 0; c < 8; ++c) {
                all_in &= corner_in[c]; // all_in becomes false if a corner is outside -> a contradiction
                all_out &= !corner_in[c]; // all_out becomes false if a corner is inside -> a contradiction
            }

            if (all_in) {
                elem_vol_frac(elem_gid) = 1.0;
                // If all of the corners are inside -> the elems is full
                return; // No more sampling is needed
            }

            if (all_out) {
                elem_vol_frac(elem_gid) = 0.0;
                // If all of the corners are outside -> the elems is empty
                return; // No more sampling is needed
            }

            double inside = 0; // Initializes the inside count for elems

            const double delta_x = 2.0 / m;
            const double delta_y = 2.0 / m;
            const double delta_z = 2.0 / m;
            double landing_pt[3];  // all in [-1:1]

            // throwing partices in xi, eta, zeta (px,py,pz)
            for (int px = 0; px < m; ++px) { // Samples m subpoints along x
                for (int py = 0; py < m; ++py) { // Samples m subpoints along y
                    for (int pz = 0; pz < m; ++pz) { // Samples m subpoints along z                           

                        landing_pt[0] = (px + 0.5) * delta_x - 1.0;
                        landing_pt[1] = (py + 0.5) * delta_y - 1.0;
                        landing_pt[2] = (pz + 0.5) * delta_z - 1.0;

                        size_t hits = 0; // Counts the ray-triangle intersections

                        bool edge_detected = false; // Highlight whether or not the ray intersects near an edge of a triangle

                        // Loops over all of the triangles in the stl
                        for (size_t tri = 0; tri < n_tris; ++tri) {

                            // see if particle landing at vertex of reference element is inside CAD part
                            if (!RayIntersectsAxisAlignedBox(landing_pt,
                                                        DIR,
                                                        &ref_cell_tri_min(elem_gid, tri, 0),
                                                        &ref_cell_tri_max(elem_gid, tri, 0))) {
                                                        continue;
                            } // Skips the triangle if the ray doesn't intersect the bounding box 

                            bool is_edge = false; // Temporarily highlight whether or not the intersection was near an edge
                            if (intersects_with_edge_detect(landing_pt,
                                                            DIR,
                                                            &ref_cell_verts_in_tri(elem_gid, tri, 0, 0),
                                                            &ref_cell_verts_in_tri(elem_gid, tri, 1, 0),
                                                            &ref_cell_verts_in_tri(elem_gid, tri, 2, 0),
                                                            is_edge)){
                                // Count it as an intersection
                                ++hits;
                                // If ray-triangle intersection was near an edge highlight it for a second go
                                if (is_edge) edge_detected = true;
                            } // If a ray hits increment the count

                        } // end loop over all triangles in STL file


                        if (edge_detected) { // If ray-triangle intersection was near an edge we will recast the ray
                            hits = 0; // Resets the hit counter
                            for (size_t tri = 0; tri < n_tris; ++tri) { // Loops through all triangles again using an alternate direction
                                bool dummy = false; // Dummy variable
                                if (intersects_with_edge_detect(landing_pt,
                                                            DIR,
                                                            &ref_cell_verts_in_tri(elem_gid, tri, 0, 0),
                                                            &ref_cell_verts_in_tri(elem_gid, tri, 1, 0),
                                                            &ref_cell_verts_in_tri(elem_gid, tri, 2, 0),
                                                            dummy))
                                {
                                    // Performs ray-triangle intersection using an alternate direction
                                    ++hits; // Counts the intersection
                                }
                            }
                        }

                        if (hits & 1) ++inside; // If there is an odd number of hits, the point is inside the mesh

                    } // end pz
                } // end py
            } // end px

            elem_vol_frac(elem_gid) = inside / (m * m * m); // Stores the volume fraction for the elems
        }); // end parallel for
        
        Kokkos::fence(); // guarantees that all of the parallel Kokkos operations are complete before accessing the views
        elem_vol_frac.update_host();

        auto time_2 = std::chrono::high_resolution_clock::now();

        // count the number of particles that fell inside the stl
        double lsum;
        double total_inside = 0;
        FOR_REDUCE_SUM(elem_gid, 0, num_elems, lsum, {
            lsum += elem_vol_frac(elem_gid) * (m * m * m);
        }, total_inside);

        double total_particles = ((double)num_elems) * m * m * m; // The total number of sampling points across all of the elemss at the resolution m^3

        // second timer
        std::chrono::duration <double, std::milli> ms = time_2 - time_1;
        std::cout << "\n=== Particle Statistics for m = " << int(m) << " ===\n"; // Prints the start of the particle statistics for mxmxm
        std::cout << "Total Particles      : " << int(total_particles) << "\n"; // Prints how many particles there are total
        std::cout << "Particles Inside STL : " << int(total_inside) << "\n"; // Prints how many particles are inside the stl
        std::cout << std::setprecision(10) // setprecision(n) -> n digits after the decimal point
            << "Volume Fraction      : " << 1.0 * total_inside / total_particles << "\n"; // Prints the volume fraction
        std::cout << "runtime of all tests = " << ms.count() << "ms\n";

        const size_t pts_x = Nx + 1, pts_y = Ny + 1, pts_z = Nz + 1; // The total number of grid mesh nodes in every direction
        const size_t num_points = (pts_x) * (pts_y) * (pts_z); // The total number of vertices
        const size_t elems_size = num_elems * 9; // Every cell has 8 indices and 1 leading counter, for storage of values 

        std::ostringstream fname; // Creates the file name
        fname << "elems_m" << m << ".vtk"; // Sets the file name for the given m
        FILE* fout = fopen(fname.str().c_str(), "w"); // Opens the ouput file
        if (!fout) { std::perror("fopen"); return 2; } // Checks if there is an error when the file opens

        fprintf(fout, "# vtk DataFile Version 2.0\ncell volume fractions m=%f\nASCII\n", m); // Writes the VTK header
        fprintf(fout, "DATASET UNSTRUCTURED_GRID\n"); // Declares an unstructured grid format
        fprintf(fout, "POINTS %zu float\n", num_points); // Declares the number of points

        for (int kz = 0; kz < pts_z; ++kz) // Loops over all of the points along z
            for (int jy = 0; jy < pts_y; ++jy) // Loops over all of the points along y
                for (int ix = 0; ix < pts_x; ++ix) // Loops over all of the points along x
                    fprintf(fout, "%f %f %f\n", ix * hx, jy * hy, kz * hz); // Outputs the coordinates of the points

        fprintf(fout, "\nCELLS %zu %zu\n", num_elems, elems_size); //Begins the section for elems
        auto nid = [&](int i, int j, int k) { return k * pts_y * pts_x + j * pts_x + i; }; // Computes the global index for point i,j,k
        
        for (int kz = 0; kz < Nz; ++kz) {// Loop over elemss along z
            for (int jy = 0; jy < Ny; ++jy) {// Loop of over elemss along y
                for (int ix = 0; ix < Nx; ++ix) { // Loop over elemss along x
                    const int p0 = nid(ix, jy, kz); // Vertex at corner 0, 0, 0
                    const int p1 = nid(ix + 1, jy, kz); // Vertex at corner 1, 0, 0
                    const int p2 = nid(ix + 1, jy + 1, kz); // Vertex at corner 1, 1, 0
                    const int p3 = nid(ix, jy + 1, kz); // Vertex at corner 0, 1, 0
                    const int p4 = nid(ix, jy, kz + 1); // Vertex at corner 0, 0, 1
                    const int p5 = nid(ix + 1, jy, kz + 1); // Vertex at corner 1, 0, 1
                    const int p6 = nid(ix + 1, jy + 1, kz + 1); // Vertex at corner 1, 1, 1
                    const int p7 = nid(ix, jy + 1, kz + 1); // Vertex at corner 0, 1, 1
                    fprintf(fout, "8 %d %d %d %d %d %d %d %d\n", p0, p1, p2, p3, p4, p5, p6, p7); // Writes the indices
                }
            }
        }
        fprintf(fout, "\nCELL_TYPES %zu\n", num_elems); // Begin of the cell type section
        for (double c = 0; c < num_elems; ++c) fprintf(fout, "12\n"); // VTK_HEXAHEDRON

        fprintf(fout, "\nCELL_DATA %zu\nSCALARS volume_fraction double 1\nLOOKUP_TABLE default\n", num_elems); // Writes a scalar data header
        for (size_t elem_gid = 0; elem_gid < num_elems; ++elem_gid) // Loops over elemss along z
            fprintf(fout, "%f\n", elem_vol_frac.host(elem_gid)); // Writes the volume fraction of each elems

        fclose(fout); // Closes the generated VTK file
        std::cout << "Created " << fname.str() << '\n'; // Prints to the console that it wrote a file for m = whatever (runs from 1 to what is defined on line 179)
    }
}

    Kokkos::finalize(); // Shut down Kokkos
    return 0;
}