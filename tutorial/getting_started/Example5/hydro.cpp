// -----------------------------------------------------------------------------
// 1D Hydrodynamics Simulation using MATAR and Kokkos
// -----------------------------------------------------------------------------
//
// This example demonstrates how to use MATAR to solve 1D hydrodynamics problems. It implements a
// finite element method for simulating compressible fluid flow.
//
// Key Features:
// - Uses MATAR's CArrayDual for performance-portable arrays
// - Implements a Runge-Kutta time integration scheme
// - Includes shock capturing with artificial viscosity
// - Supports multiple initial conditions (Sod shock tube, Sedov blast wave)
// - Visualizes results using ASCII plotting
//
// The simulation solves the following conservation equations:
// 1. Mass conservation
// 2. Momentum conservation
// 3. Energy conservation
//
// The code uses a cell-centered finite element approach where:
// - Cells store density, pressure, internal energy, and sound speed
// - Nodes store velocity and position
// - Corners (cell-node connections) store forces and mass
//
// Written by: Nathaniel Morgan
//   Mar 8, 2022
// Updated by: Jacob Moore
//   Jun 13, 2025
//
// Usage:
//   ./hydro --kokkos-threads=4  # Run with 4 threads
//
// OpenMP Settings (if using OpenMP backend):
//   setenv OMP_PROC_BIND true
//   setenv OMP_PROC_BIND spread
//   setenv OMP_PLACES threads
//   setenv OMP_NUM_THREADS 2
// -----------------------------------------------------------------------------

#include "matar.h"
#include <stdio.h>
#include <math.h>  // c math lib
#include <chrono>

// -----------------------------------------------------------------------------
//    Global variables
// -----------------------------------------------------------------------------
const double fuzz = 1.0E-15;  // Small number to prevent division by zero
const double huge = 1.0E15;   // Large number for initialization

using namespace mtr;

// Output frequency for visualization
const int output_interval = 100;

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

// Structure to define initial conditions for different regions
struct region_t {
   double x_min;    // Left boundary of region
   double x_max;    // Right boundary of region
   double den;      // Initial density
   double sie;      // Initial specific internal energy
   double vel;      // Initial velocity
   double gamma;    // Adiabatic index (ratio of specific heats)
};

// -----------------------------------------------------------------------------
// Function Declarations
// -----------------------------------------------------------------------------

// Helper functions for mesh connectivity
KOKKOS_INLINE_FUNCTION
int get_corners_in_cell(int,int);  // Get corner indices for a cell

KOKKOS_INLINE_FUNCTION
int get_corners_in_node(int,int);  // Get corner indices for a node

// Visualization function
void plot_density_vs_position(const CArrayDual<double>& cell_coords, 
                            const CArrayDual<double>& cell_den,
                            const int num_cells,
                            const char* title);

// -----------------------------------------------------------------------------
//    The Main function
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]){
    
    // -------------------------------
    // Simulation Parameters
    // -------------------------------

    // Time integration settings
    const double time_max = 20.0;     // End time of simulation
    double       dt       = 0.01;     // Initial time step
    const double dt_max   = 1.0;      // Maximum allowed time step
    const double dt_cfl   = 0.3;      // CFL number for stability
    const int    num_rk_stages = 2;   // Number of Runge-Kutta stages
    const int    max_cycles = 2000000;// Maximum number of time steps

    // Mesh settings
    const double  x_min = 0.0;        // Left boundary of domain
    const double  x_max = 100.0;      // Right boundary of domain
    const int num_cells = 100000;     // Number of cells in mesh

    // Physics parameters
    const double sspd_min = 1.0E-3;   // Minimum sound speed for stability

    // -------------------------------
    // Initial Conditions
    // -------------------------------
    
    // Sod Shock Tube Problem
    // This is a standard test case for compressible flow solvers
    // It consists of two regions with different densities and pressures
    // separated by a diaphragm that is removed at t=0
    const int num_regions = 2;
    region_t ics[num_regions];
    
    // High pressure region (left)
    ics[0].x_min = 0.0;
    ics[0].x_max = 50.0;
    ics[0].den   = 1.0;    // Higher density
    ics[0].sie   = 2.5;    // Higher internal energy
    ics[0].vel   = 0.0;    // Initially at rest
    ics[0].gamma = 1.4;    // Ideal gas constant
    
    // Low pressure region (right)
    ics[1].x_min = 50.0;
    ics[1].x_max = 100.0;
    ics[1].den   = 0.125;  // Lower density
    ics[1].sie   = 2.0;    // Lower internal energy
    ics[1].vel   = 0.0;    // Initially at rest
    ics[1].gamma = 1.4;    // Ideal gas constant

    // Alternative: Sedov Blast Wave Problem
    // Uncomment to use instead of Sod problem
    // This test case simulates a point explosion in a uniform medium
    /*
    const int num_regions = 2;
    region_t ics[num_regions];
    
    ics[0].x_min = 0.0;
    ics[0].x_max = 1.0;
    ics[0].den   = 1.0;
    ics[0].sie   = 100.0;  // High energy in small region
    ics[0].vel   = 0.0;
    ics[0].gamma = 1.4;
    
    ics[1].x_min = 1.0;
    ics[1].x_max = 100.0;
    ics[1].den   = 1.0;
    ics[1].sie   = 0.1;    // Low energy in surrounding region
    ics[1].vel   = 0.0;
    ics[1].gamma = 1.4;
    */

    // -------------------------------
    // Initialize MATAR and Kokkos
    // -------------------------------
    printf("\nstarting FE code\n");
    
    // This is the meat in the code, it must be inside Kokkos scope
    MATAR_INITIALIZE();
    {
        
        // 1D linear element int( -Grad(phi) dot jJ^{-1} )
        const double integral_grad_basis[2] = {1.0, -1.0};
        
        // calculate mesh information based on inputs
        const int num_nodes = num_cells+1;
        const int num_corners = 2*num_cells;
        double dx = (x_max-x_min)/num_cells;
        
        // initialize the time to zero
        double time = 0.0;
    
        
        // --- setup variables based on user inputs ---
        
        // cell variables
        CArrayDual<double> cell_den(num_cells);   // density
        CArrayDual<double> cell_pres(num_cells);  // pressure
        CArrayDual<double> cell_sspd(num_cells);  // sound speed
        CArrayDual<double> cell_sie(num_cells);   // specific internal energy
        CArrayDual<double> cell_sie_n(num_cells); // specific internal energy at t_n
        CArrayDual<double> cell_mass(num_cells);  // mass in the cell
        
        CArrayDual<double> cell_gamma(num_cells);  // gamma law gas
        
        // nodal variables
        CArrayDual<double> node_vel(num_nodes);    // velocity
        CArrayDual<double> node_vel_n(num_nodes);  // the velocity at t_n
        CArrayDual<double> node_mass(num_nodes);   // mass of node
        
        // corner variables
        CArrayDual<double> corner_force(num_corners); // force from cell to node
        CArrayDual<double> corner_mass(num_corners);  // mass in cell corner
        CArrayDual<double> corner_vel(num_corners);   // velocity in cell corner
        
        // mesh variables
        CArrayDual<double> cell_coords(num_cells);   // coordinates of cell
        CArrayDual<double> cell_vol(num_cells);      // volume of the cell
        
        CArrayDual<double> node_coords(num_nodes);   // coordinates of nodes
        CArrayDual<double> node_coords_n(num_nodes); // coordinates at t_n
        
        
        
        // --- build the mesh ---
        
        // calculate nodal coordinates of the mesh
        FOR_ALL (node_id, 0, num_nodes, {
           node_coords(node_id) = double(node_id) * dx;
        }); // end parallel for on device

        
        
        // calculate cell center coordinates of the mesh
        FOR_ALL (cell_id, 0, num_cells, {
            cell_coords(cell_id) =
                           0.5*(node_coords(cell_id) + node_coords(cell_id+1));
            
            cell_vol(cell_id)  = node_coords(cell_id+1) - node_coords(cell_id);
        }); // end parallel for on device

        
        
        // --- initial state on the mesh ---
        
        // initial cell state
        FOR_ALL (cell_id, 0, num_cells, {
            
            // loop over the regions
            for (int reg=0; reg<num_regions; reg++){

               if (cell_coords(cell_id) >= ics[reg].x_min &&
                   cell_coords(cell_id) <= ics[reg].x_max){
                     
                   cell_den(cell_id) = ics[reg].den;
                   cell_sie(cell_id) = ics[reg].sie;
                   cell_gamma(cell_id) = ics[reg].gamma;
                   
                   cell_pres(cell_id)  = cell_den(cell_id)*cell_sie(cell_id)*
                                         (cell_gamma(cell_id) - 1.0);
                   
                   cell_sspd(cell_id)  = sqrt( cell_gamma(cell_id)*
                                         cell_pres(cell_id)/cell_den(cell_id) );
                   
                   cell_mass(cell_id)  = ics[reg].den*cell_vol(cell_id);
                   
                   // get the corners id's
                   int corner_id_0 = get_corners_in_cell(cell_id, 0); // left
                   int corner_id_1 = get_corners_in_cell(cell_id, 1); // right
                   
                   // scatter mass to the cell corners
                   corner_mass(corner_id_0) = 0.5*cell_mass(cell_id);
                   corner_mass(corner_id_1) = 0.5*cell_mass(cell_id);
                   
                   // scatter velocity to the cell corners
                   corner_vel(corner_id_0) = ics[reg].vel;
                   corner_vel(corner_id_1) = ics[reg].vel;
                   
               } // end if
            } // end for
        }); // end parallel for on device
        
        
        // intialize the nodal state that is internal to the mesh
        FOR_ALL (node_id, 1, num_nodes-1, {
            node_mass(node_id) = 0.0;
            node_vel(node_id) = 0.0;
            
            for (int corner_lid=0; corner_lid<2; corner_lid++){
                // get the global index for the corner
                int corner_gid = get_corners_in_node(node_id, corner_lid);
                
                node_mass(node_id) += corner_mass(corner_gid); // tally mass
                node_vel(node_id) += 0.5*corner_vel(corner_gid); // average
            } // end for
        }); // end parallel for on device

        
        // calculate boundary nodal masses and set boundary velocities
        RUN ({
            node_mass(0) = corner_mass(0);
            node_mass(num_nodes-1) = corner_mass(num_corners-1);
            
            node_vel(0) = 0.0;
            node_vel(num_nodes-1) = 0.0;
        }); // end run once on the device

        
        // update the host side to print (i.e., copy from device to host)
        cell_coords.update_host();
        cell_den.update_host();
        cell_pres.update_host();
        cell_sie.update_host();
        node_vel.update_host();
        
        // Plot initial state
        plot_density_vs_position(cell_coords, cell_den, num_cells, "Initial State:");
        
       
        // total energy check
        double total_e = 0.0;
        double e_lcl = 0.0;
        FOR_REDUCE_SUM(cell_id, 0, num_cells, e_lcl, {
               e_lcl += cell_mass(cell_id)*cell_sie(cell_id) +
                        0.5*cell_mass(cell_id)*0.5*pow(node_vel(cell_id), 2) +
                        0.5*cell_mass(cell_id)*0.5*pow(node_vel(cell_id+1), 2);
        }, total_e);
        
        
        auto time_1 = std::chrono::high_resolution_clock::now();
        
        // -------------------------------------
        // Main Time Integration Loop
        // -------------------------------------
        for (int cycle = 0; cycle<max_cycles; cycle++){
           
            // Calculate time step based on CFL condition
            // dt = CFL * dx / (sound_speed + |velocity|)
            double dt_ceiling = dt*1.1;
            
            // Parallel reduction to find minimum dt across all cells
            double dt_lcl;
            double min_dt_calc;
            FOR_REDUCE_MIN(cell_id, 0, num_cells, dt_lcl, {
                // mesh size
                double dx_lcl = node_coords(cell_id+1) - node_coords(cell_id);
                
                // local dt calc
                double dt_lcl_ = dt_cfl*dx_lcl/(cell_sspd(cell_id) + fuzz);
                
                // make dt be in bounds
                dt_lcl_ = fmin(dt_lcl_, dt_max);
                dt_lcl_ = fmin(dt_lcl_, time_max-time);
                dt_lcl_ = fmin(dt_lcl_, dt_ceiling);
        
                if (dt_lcl_ < dt_lcl) dt_lcl = dt_lcl_;
                        
            }, min_dt_calc); // end parallel reduction on min
            MATAR_FENCE();
            
            // Update time step
            if(min_dt_calc < dt) dt = min_dt_calc;
            
            if (dt<=fuzz) break;
            
            // -------------------------------
            // Runge-Kutta Time Integration
            // -------------------------------
            for (int rk_stage = 0; rk_stage < num_rk_stages; rk_stage++ ){
               
                // Calculate RK coefficient
                double rk_alpha = 1.0/(double(num_rk_stages) - double(rk_stage));
            
                // Save state at beginning of time step
                if (rk_stage == 0){
                    
                    // nodal state
                    FOR_ALL (node_id, 0, num_nodes, {
                        node_vel_n(node_id)    = node_vel(node_id);
                        node_coords_n(node_id) = node_coords(node_id);
                    }); // end parallel for on device
                    
                    
                    // cell state
                    FOR_ALL (cell_id, 0, num_cells, {
                        cell_sie_n(cell_id) = cell_sie(cell_id);
                    }); // end parallel for on device

                } // end if
                
                // -------------------------------
                // Calculate Forces and Update State
                // -------------------------------
                
                // 1. Calculate corner forces including artificial viscosity
                //    for shock capturing
                FOR_ALL (cell_id, 0, num_cells, {
                
                    double visc;
                    double visc_HO;
                    
                    // solve Riemann problem in compression
                    double dvel = node_vel(cell_id+1) - node_vel(cell_id);
                    if (dvel < 0.0){
                       
                        // first-order dissipation from Riemann problem
                        visc =
                           1.0/2.0*cell_den(cell_id)*
                                   cell_sspd(cell_id)*fabs(dvel) +
                           1.2/4.0*cell_den(cell_id)*pow(dvel, 2.0);
                       
                        // higher-order dissipation, can be Order(h^N)
                        visc_HO = 0.0;
                        
                    }
                    else {
                        
                        // dissipation from Riemann problem Order h^2
                        visc = -1.2/4.0*cell_den(cell_id)*pow(dvel, 2.0);
                       
                        // higher-order dissipation, can be Order(h^N)
                        visc_HO = 0.0;
                        
                    } // end if
                    
                    // apply the viscosity only in regions near a shock
                    // visc_limited = alpha*visc where alpha=0 is smooth flow
                    // and alpha=1 is a shock
                    // alpha = coef*abs(delta vel)/sound_speed
                    //      set coeff < 1 (but > 0) to bias limiter towards
                    //      high-order dissipation
                    //      set coeff > 1 to bias limiter towards low-order
                    //      dissipation
                    double ratio = 20.0*fabs(dvel)/(cell_sspd(cell_id)+fuzz);
                    double alpha = fmax(0.0, fmin(1.0, ratio));
                    
                    // apply shock detector to viscosity
                    visc = alpha*visc + (1.0-alpha)*visc_HO;
                    
                    // get the corners id's
                    int corner_id_0 = get_corners_in_cell(cell_id, 0); // left
                    int corner_id_1 = get_corners_in_cell(cell_id, 1); // right
                    
                    
                    // left corner force
                    corner_force(corner_id_0) =
                             integral_grad_basis[0]*(-cell_pres(cell_id)-visc);
                    
                    // right corner force
                    corner_force(corner_id_1) =
                             integral_grad_basis[1]*(-cell_pres(cell_id)-visc);
                    
                }); // end parallel for on device
                
                
                
                // 2. Update velocities using forces
                FOR_ALL (node_id, 1, num_nodes-1, {
                    
                    // get the global indices for the corners of this node
                    int corner_id_0 = get_corners_in_node(node_id, 0); // left
                    int corner_id_1 = get_corners_in_node(node_id, 1); // right
                    
                    double force_tally = corner_force(corner_id_0) +
                                         corner_force(corner_id_1);
                    
                    // TODO: update velocity: node_vel_{n+1} = node_vel_n + alpha*dt/node_mass*force_tally
                    
                    
                }); // end parallel for on device

                
                // applying a wall BC on velocity
                RUN ({
                    node_vel(0) = 0.0;
                    node_vel(num_nodes-1) = 0.0;
                });  // end run once on the device

                
                // Warning: free surface BC requires including node_vel calc,
                //   node_vel(node_id) = node_vel_n(node_id) +
                //       rk_alpha*dt/node_mass(node_id)*corner_force(0 or last);
                
                
                // 3. Update mesh positions
                FOR_ALL (node_id, 0, num_nodes, {
                    
                    // velocity at t+1/2
                    double half_vel = 0.5*(node_vel_n(node_id) +
                                           node_vel(node_id));
                    
                    // TODO: update coordinates: node_coords_{n+1} = node_coords_n + alpha*dt*half_vel
                    
                    
                }); // end parallel for on device
                
                
                
                // --- Calculate new internal energy ---
                
                // e_new = e_n + alpha*dt/mass*Sum(forces*vel_half)
                FOR_ALL (cell_id, 0, num_cells, {
                    
                    // get the global indices for the corners of this cell
                    int corner_id_0 = get_corners_in_cell(cell_id, 0); // left
                    int corner_id_1 = get_corners_in_cell(cell_id, 1); // right
                    
                    double half_vel_0 = 0.5*(node_vel_n(cell_id) +
                                             node_vel(cell_id));
                    double half_vel_1 = 0.5*(node_vel_n(cell_id+1) +
                                             node_vel(cell_id+1));
                    
                    double power_tally = corner_force(corner_id_0)*half_vel_0 +
                                         corner_force(corner_id_1)*half_vel_1;
                    
                    // update specific interal energy
                    cell_sie(cell_id) = cell_sie_n(cell_id) -
                                    rk_alpha*dt/cell_mass(cell_id)*power_tally;
                    
                    
                    // --- Update the state on the mesh ---
                    // update vol
                    cell_vol(cell_id)  = node_coords(cell_id+1) -
                                         node_coords(cell_id);
                    
                    // TODO: update coordinates
                    cell_coords(cell_id) = 0.5*(node_coords(cell_id) +
                                                node_coords(cell_id+1));
                    
                    // update density
                    cell_den(cell_id) = cell_mass(cell_id)/cell_vol(cell_id);
                    
                    // update pressure via EOS
                    cell_pres(cell_id) = cell_den(cell_id)*cell_sie(cell_id)*
                                         (cell_gamma(cell_id) - 1.0);
                    
                    // update sound speed
                    cell_sspd(cell_id) = sqrt( cell_gamma(cell_id)*
                                         cell_pres(cell_id)/cell_den(cell_id) );
                    cell_sspd(cell_id) = fmax(cell_sspd(cell_id), sspd_min);
                    
                }); // end parallel for on device
                // MATAR_FENCE();
                
                
            } // end rk loop

            // -------------------------------
            // Output and Visualization
            // -------------------------------
            if (cycle % output_interval == 0) {
                std::string title = "State at cycle " + std::to_string(cycle);
                cell_coords.update_host();
                cell_den.update_host();
                plot_density_vs_position(cell_coords, cell_den, num_cells, title.c_str());
            }
            
            // Update simulation time
            time += dt;
            if (abs(time-time_max)<=fuzz) time=time_max;
            
        } // end time integration loop
        //------------- Done with calculation ------------------
        MATAR_FENCE();
        
        auto time_2 = std::chrono::high_resolution_clock::now();
        
        auto calc_time = std::chrono::duration_cast
                           <std::chrono::nanoseconds>(time_2 - time_1).count();
        
        
        // -------------------------------
        //    Print final state 
        // -------------------------------
        
        // update the host side to print (i.e., copy from device to host)
        cell_coords.update_host();
        cell_den.update_host();
        cell_pres.update_host();
        cell_sie.update_host();
        node_vel.update_host();
        

        
        // Plot final state
        plot_density_vs_position(cell_coords, cell_den, num_cells, "Final State:");

        printf("\nCalculation time in seconds: %f \n", calc_time * 1e-9);

    
        // total energy check
        double total_e_final = 0.0;
        double ef_lcl = 0.0;
        FOR_REDUCE_SUM(cell_id, 0, num_cells, ef_lcl, {
               ef_lcl += cell_mass(cell_id)*cell_sie(cell_id) +
                         0.5*cell_mass(cell_id)*0.5*pow(node_vel(cell_id), 2) +
                         0.5*cell_mass(cell_id)*0.5*pow(node_vel(cell_id+1), 2);
        }, total_e_final);
        MATAR_FENCE();
        
        printf("total energy, TE(t=0) = %f", total_e);
        printf(" , TE(t=end) = %f" ,  total_e_final);
        printf(" , TE error = %f \n", total_e_final-total_e);
        
        
        // ======== Done using Kokkos ============
        
    } // end of kokkos scope
    MATAR_FINALIZE();
    
    
    printf("\nfinished\n\n");
    return 0;
  
} // end main function



// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// Get the global index of a corner within a cell
// cell_gid: Global cell index
// corner_lid: Local corner index (0=left, 1=right)
KOKKOS_INLINE_FUNCTION
int get_corners_in_cell(int cell_gid, int corner_lid){
    return (2*cell_gid + corner_lid);
}

// Get the global index of a corner within a node
// node_gid: Global node index
// corner_lid: Local corner index (0=left, 1=right)
KOKKOS_INLINE_FUNCTION
int get_corners_in_node(int node_gid, int corner_lid){
    return (2*node_gid - 1 + corner_lid);
}

// -----------------------------------------------------------------------------
// Visualization Functions
// -----------------------------------------------------------------------------

// Plot density vs position using ASCII art
// This function creates a simple visualization of the density field
// by plotting it as a function of position using ASCII characters.
//
// Parameters:
//   cell_coords: Array of cell center coordinates
//   cell_den: Array of cell densities
//   num_cells: Number of cells in the mesh
//   title: Title to display above the plot
void plot_density_vs_position(const CArrayDual<double>& cell_coords, 
                            const CArrayDual<double>& cell_den,
                            const int num_cells,
                            const char* title) {
    // Plot dimensions and style
    const int PLOT_WIDTH = 120;   // Width of plot in characters
    const int PLOT_HEIGHT = 20;   // Height of plot in characters
    const char PLOT_CHAR = '*';   // Character used for plotting
    
    // Find data range for scaling
    double min_x = cell_coords.host(0);
    double max_x = cell_coords.host(num_cells-1);
    double min_den = cell_den.host(0);
    double max_den = cell_den.host(0);
    
    // Calculate min/max density for scaling
    for(int i = 0; i < num_cells; i++) {
        min_den = std::min(min_den, cell_den.host(i));
        max_den = std::max(max_den, cell_den.host(i));
    }
    
    // Add padding to density range for better visualization
    double den_range = max_den - min_den;
    min_den -= 0.1 * den_range;
    max_den += 0.1 * den_range;
    
    // Print plot header
    printf("\n%s\n", title);
    printf("Density vs Position\n");
    printf("Density range: %.3f to %.3f\n", min_den, max_den);
    printf("Position range: %.1f to %.1f\n\n", min_x, max_x);
    
    // Create the plot row by row
    for(int row = PLOT_HEIGHT-1; row >= 0; row--) {
        // Calculate density value for this row
        double den_value = min_den + (max_den - min_den) * row / (PLOT_HEIGHT-1);
        printf("%8.3f |", den_value);
        
        // Plot each column
        for(int col = 0; col < PLOT_WIDTH; col++) {
            double x_value = min_x + (max_x - min_x) * col / (PLOT_WIDTH-1);
            bool point_plotted = false;
            
            // Find and plot the closest cell to this x position
            for(int i = 0; i < num_cells; i++) {
                if(std::abs(cell_coords.host(i) - x_value) < (max_x - min_x)/(2*PLOT_WIDTH)) {
                    if(cell_den.host(i) >= den_value) {
                        printf("%c", PLOT_CHAR);
                        point_plotted = true;
                        break;
                    }
                }
            }
            if(!point_plotted) printf(" ");
        }
        printf("\n");
    }
    
    // Print x-axis
    printf("         +");
    for(int i = 0; i < PLOT_WIDTH; i++) printf("-");
    printf("\n");
    printf("          ");
    printf("%.1f", min_x);
    for(int i = 0; i < PLOT_WIDTH-8; i++) printf(" ");
    printf("%.1f\n\n", max_x);
    
    // Ensure plot is displayed immediately
    fflush(stdout);
}



