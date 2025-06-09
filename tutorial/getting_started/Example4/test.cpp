// fem3d_single_element.cpp - Standalone C++ FEM solver for a single 8-node hexahedral element (large deformation)

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

constexpr int ndof = 24; // 8 nodes * 3 DOF per node
constexpr int nnodes = 8;

// Initial nodal coordinates for the reference element
// Each node has (x, y, z) coordinates
double coords[nnodes][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
};

// Deformed nodal coordinates after applying deformation gradient F
double updated_coords[nnodes][3];

// Applies deformation gradient F to the reference coordinates
double updated_coords[nnodes][3];
// F is a 3x3 deformation gradient tensor
void apply_deformation(const double F[3][3]) {
    for (int i = 0; i < nnodes; ++i)
        for (int j = 0; j < 3; ++j) {
            updated_coords[i][j] = 0.0;
            for (int k = 0; k < 3; ++k)
                updated_coords[i][j] += F[j][k] * coords[i][k];
        }
}

// Computes the shape function derivatives in the parent (xi, eta, zeta) space
// Inputs: xi, eta, zeta - local coordinates of the Gauss point
// Output: dN_dxi[i][j] - derivative of shape function i with respect to local direction j
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

// Computes inverse of a 3x3 matrix A, stores in invA
// Returns true if successful, false if matrix is singular
bool invert3x3(const double A[3][3], double invA[3][3]) {
    double det =
        A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) -
        A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) +
        A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);

    if (fabs(det) < 1e-12) return false;
    double idet = 1.0 / det;

    invA[0][0] =  (A[1][1]*A[2][2] - A[1][2]*A[2][1]) * idet;
    invA[0][1] = -(A[0][1]*A[2][2] - A[0][2]*A[2][1]) * idet;
    invA[0][2] =  (A[0][1]*A[1][2] - A[0][2]*A[1][1]) * idet;

    invA[1][0] = -(A[1][0]*A[2][2] - A[1][2]*A[2][0]) * idet;
    invA[1][1] =  (A[0][0]*A[2][2] - A[0][2]*A[2][0]) * idet;
    invA[1][2] = -(A[0][0]*A[1][2] - A[0][2]*A[1][0]) * idet;

    invA[2][0] =  (A[1][0]*A[2][1] - A[1][1]*A[2][0]) * idet;
    invA[2][1] = -(A[0][0]*A[2][1] - A[0][1]*A[2][0]) * idet;
    invA[2][2] =  (A[0][0]*A[1][1] - A[0][1]*A[1][0]) * idet;
    return true;
}

// Solves the linear system Ax = b using Gaussian elimination
// A is overwritten; b is input RHS, x is the solution vector
void solve_gauss(double A[ndof][ndof], double* b, double* x) {
    for (int i = 0; i < ndof; ++i) x[i] = b[i];
    for (int i = 0; i < ndof; ++i) {
        double pivot = A[i][i];
        for (int j = i; j < ndof; ++j) A[i][j] /= pivot;
        x[i] /= pivot;

        for (int k = i + 1; k < ndof; ++k) {
            double f = A[k][i];
            for (int j = i; j < ndof; ++j) A[k][j] -= f * A[i][j];
            x[k] -= f * x[i];
        }
    }
    for (int i = ndof - 1; i >= 0; --i)
        for (int k = 0; k < i; ++k) {
            double f = A[k][i];
            x[k] -= f * x[i];
        }
}

// Main driver for single element finite element simulation
int main() {
    // Deformation gradient tensor F
    double F[3][3] = {{1.2, 0.1, 0.0},
                      {0.0, 1.1, 0.0},
                      {0.0, 0.0, 1.3}};
    apply_deformation(F);

    // Ke: Element stiffness matrix (24x24)
    // fe: Element force vector (24x1)
    // ue: Solution displacement vector (24x1)
    double Ke[ndof][ndof] = {{0}};
    double fe[ndof] = {0};
    double ue[ndof] = {0};

    double mu = 80e9;      // Shear modulus
    double lambda = 120e9; // First Lamé parameter (unused here)

    const double gp[] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)}; // 2-point Gauss quadrature

    // Loop over 8 Gauss points
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
    for (int k = 0; k < 2; ++k) {
        double xi = gp[i], eta = gp[j], zeta = gp[k];

        // dN_dxi: shape function derivatives in reference space
        double dN_dxi[8][3];
        shape_function_derivatives(xi, eta, zeta, dN_dxi);

        // Compute Jacobian matrix J = dX/dxi
        double J[3][3] = {{0}};
        for (int a = 0; a < 8; ++a)
            for (int r = 0; r < 3; ++r)
                for (int s = 0; s < 3; ++s)
                    J[r][s] += dN_dxi[a][r] * coords[a][s];

        // Inverse Jacobian invJ = dxi/dX
        double invJ[3][3];
        if (!invert3x3(J, invJ)) continue;

        // dN_dx_ref: shape function gradients w.r.t reference coordinates
        double dN_dx_ref[8][3] = {{0}};
        for (int a = 0; a < 8; ++a)
            for (int r = 0; r < 3; ++r)
                for (int s = 0; s < 3; ++s)
                    dN_dx_ref[a][r] += invJ[r][s] * dN_dxi[a][s];

        // F_local: Deformation gradient at Gauss point
        double F_local[3][3] = {{0}};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int a = 0; a < 8; ++a)
                    F_local[i][j] += updated_coords[a][i] * dN_dx_ref[a][j];

        // Determinant of deformation gradient
        double detF = F_local[0][0]*(F_local[1][1]*F_local[2][2] - F_local[1][2]*F_local[2][1]) -
                      F_local[0][1]*(F_local[1][0]*F_local[2][2] - F_local[1][2]*F_local[2][0]) +
                      F_local[0][2]*(F_local[1][0]*F_local[2][1] - F_local[1][1]*F_local[2][0]);

        if (detF <= 0) continue;

        // Reference configuration Jacobian Jref
        double Jref[3][3] = {{0}};
        for (int a = 0; a < 8; ++a)
            for (int r = 0; r < 3; ++r)
                for (int s = 0; s < 3; ++s)
                    Jref[r][s] += dN_dxi[a][r] * coords[a][s];

        double detJ = Jref[0][0]*(Jref[1][1]*Jref[2][2] - Jref[1][2]*Jref[2][1]) -
                      Jref[0][1]*(Jref[1][0]*Jref[2][2] - Jref[1][2]*Jref[2][0]) +
                      Jref[0][2]*(Jref[1][0]*Jref[2][1] - Jref[1][1]*Jref[2][0]);

        // dN_dx: shape function gradients w.r.t spatial coordinates
        double dN_dx[8][3] = {{0}};
        for (int a = 0; a < 8; ++a)
            for (int r = 0; r < 3; ++r)
                for (int s = 0; s < 3; ++s)
                    dN_dx[a][r] += invJ[r][s] * dN_dxi[a][s];

        // Strain-displacement matrix B
        double B[9][24] = {{0}};
        for (int a = 0; a < 8; ++a) {
            int ia = 3 * a;
            B[0][ia + 0] = dN_dx[a][0];
            B[1][ia + 1] = dN_dx[a][1];
            B[2][ia + 2] = dN_dx[a][2];
            B[3][ia + 1] = dN_dx[a][2];
            B[3][ia + 2] = dN_dx[a][1];
            B[4][ia + 0] = dN_dx[a][2];
            B[4][ia + 2] = dN_dx[a][0];
            B[5][ia + 0] = dN_dx[a][1];
            B[5][ia + 1] = dN_dx[a][0];
        }

        // BtB = B^T * B
        double BtB[24][24] = {{0}};
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
                for (int m = 0; m < 3; ++m)
                    BtB[i][j] += B[m][i] * B[m][j];

        // Add contribution to element stiffness matrix
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
                Ke[i][j] += mu * BtB[i][j] * fabs(detJ);
    }

    // Apply Dirichlet boundary conditions to first 12 DOFs
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < ndof; ++j) Ke[i][j] = Ke[j][i] = 0;
        Ke[i][i] = 1.0;
        fe[i] = 0;
    }

    // Apply external force in -z direction to node 6
    fe[3*6 + 2] = -1000.0;

    // Solve linear system Ke * ue = fe
    solve_gauss(Ke, fe, ue);

    // Write results to file and update deformed coordinates
    std::ofstream out("disp.csv");
    out << "Node,u_x,u_y,u_z\n";
    for (int i = 0; i < nnodes; ++i) {
        for (int j = 0; j < 3; ++j)
            updated_coords[i][j] += ue[3*i + j];
        out << i << "," << ue[3*i] << "," << ue[3*i+1] << "," << ue[3*i+2] << "\n";
    }
    out.close();
    return 0;
}
