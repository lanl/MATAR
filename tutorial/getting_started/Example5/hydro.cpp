// -----------------------------------------------------------------------------
//
//    This is a 1D c++ finite element code for material dynamics written using
//    MATAR+Kokkos for performance portabilityover CPUs and GPUs
//
//    Written by: Nathaniel Morgan
//      Mar 8, 2022
//    Updated by: Jacob Moore
//      Jun 13, 2025
//
//    To run the code with e.g., 4 threads, type
//      ./hydro --kokkos-threads=4
//
//    openMP settings:
//      setenv OMP_PROC_BIND true
//      setenv OMP_PROC_BIND spread
//      setenv OMP_PLACES threads
//      setenv OMP_NUM_THREADS 2
// -----------------------------------------------------------------------------

#include "matar.h"
#include <stdio.h>
#include <math.h>  // c math lib
#include <chrono>

// -----------------------------------------------------------------------------
//    Global variables
// -----------------------------------------------------------------------------
const double fuzz = 1.0E-15;
const double huge = 1.0E15;

using namespace mtr;


const int output_interval = 10000000;

// -----------------------------------------------------------------------------
//    A region
// -----------------------------------------------------------------------------
struct region_t{
   double x_min;
   double x_max;
   double den;
   double sie;
   double vel;
   double gamma;
};


// -----------------------------------------------------------------------------
//    Functions
// -----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
int get_corners_in_cell(int,int);

KOKKOS_INLINE_FUNCTION
int get_corners_in_node(int,int);

// Function to plot density vs position using ASCII art
void plot_density_vs_position(const CArrayDual<double>& cell_coords, 
                            const CArrayDual<double>& cell_den,
                            const int num_cells,
                            const char* title);

// -----------------------------------------------------------------------------
//    The Main function
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]){
    
    // -------------------------------
    //    User settable variables
    // -------------------------------

    // time step settings
    const double time_max = 20.0;
    double       dt       = 0.01;
    const double dt_max   = 1.0;
    const double dt_cfl   = 0.3;
    const int    num_rk_stages = 2;
    const int    max_cycles = 2000000;

    // mesh information
    const double  x_min = 0.0;
    const double  x_max = 100.0;
    const int num_cells = 100000;

    // global model parameters
    const double sspd_min = 1.0E-3;

    // intial conditions for each region
    
    // Sod
    const int num_regions = 2;
    region_t ics[num_regions];
    
    ics[0].x_min = 0.0;
    ics[0].x_max = 50.0;
    ics[0].den   = 1.0;
    ics[0].sie   = 2.5;
    ics[0].vel   = 0.0;
    ics[0].gamma = 1.4;
    
    ics[1].x_min = 50.0;
    ics[1].x_max = 100.0;
    ics[1].den   = 0.125;
    ics[1].sie   = 2.0;
    ics[1].vel   = 0.0;
    ics[1].gamma = 1.4;


    // Sedov
    // const int num_regions = 2;
    // region_t ics[num_regions];
    
    // ics[0].x_min = 0.0;
    // ics[0].x_max = 1.0;
    // ics[0].den   = 1.0;
    // ics[0].sie   = 100.0;
    // ics[0].vel   = 0.0;
    // ics[0].gamma = 1.4;
    
    // ics[1].x_min = 1.0;
    // ics[1].x_max = 100.0;
    // ics[1].den   = 1.0;
    // ics[1].sie   = 0.1;
    // ics[1].vel   = 0.0;
    // ics[1].gamma = 1.4;

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
        
        // -------------------------------
        //    Print intiial state to file
        // -------------------------------
        
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
        // Solve equations until time=time_max
        // -------------------------------------
        for (int cycle = 0; cycle<max_cycles; cycle++){
           
            
            // get the new time step
            double dt_ceiling = dt*1.1;
            
            // parallel reduction with min
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
            
            // save the min dt
            if(min_dt_calc < dt) dt = min_dt_calc;
            
            
            //printf("time = %f, dt = %f \n", time, dt);
            if (dt<=fuzz) break;
            
            
            
            // --- integrate forward in time ---
            
            // Runge-Kutta loop
            for (int rk_stage = 0; rk_stage < num_rk_stages; rk_stage++ ){
               
                // rk coefficient on dt
                double rk_alpha = 1.0/
                                     (double(num_rk_stages) - double(rk_stage));
            
                
                // save the state at t_n
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
                
                
                
                // --- Calculate corner forces ---
                
                // force is calculated with a single point quadrature approach
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
                
                
                
                // --- Calculate new velocity ---
                
                // v_new = v_n + alpha*dt/mass*Sum(forces)
                FOR_ALL (node_id, 1, num_nodes-1, {
                    
                    // get the global indices for the corners of this node
                    int corner_id_0 = get_corners_in_node(node_id, 0); // left
                    int corner_id_1 = get_corners_in_node(node_id, 1); // right
                    
                    double force_tally = corner_force(corner_id_0) +
                                         corner_force(corner_id_1);
                    
                    // update velocity
                    node_vel(node_id) = node_vel_n(node_id) +
                                rk_alpha*dt/node_mass(node_id)*force_tally;
                    
                }); // end parallel for on device

                
                // applying a wall BC on velocity
                RUN ({
                    node_vel(0) = 0.0;
                    node_vel(num_nodes-1) = 0.0;
                });  // end run once on the device

                
                // Warning: free surface BC requires including node_vel calc,
                //   node_vel(node_id) = node_vel_n(node_id) +
                //       rk_alpha*dt/node_mass(node_id)*corner_force(0 or last);
                
                
                // --- Calculate new mesh positions ---
                
                // x_new = x_n + alpha*dt*vel_half
                FOR_ALL (node_id, 0, num_nodes, {
                    
                    // velocity at t+1/2
                    double half_vel = 0.5*(node_vel_n(node_id) +
                                           node_vel(node_id));
                    
                    // update velocity
                    node_coords(node_id) = node_coords_n(node_id) +
                                           rk_alpha*dt*half_vel;
                    
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
                    
                    // update coordinates
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

            // Plot the density vs position if the cycle is a multiple of 100
            if (cycle % output_interval == 0) {
                std::string title = "State at cycle " + std::to_string(cycle);
                cell_coords.update_host();
                cell_den.update_host();
                plot_density_vs_position(cell_coords, cell_den, num_cells, title.c_str());
            }

            
            
            // update the time
            time += dt;
            if (abs(time-time_max)<=fuzz) time=time_max;
            
        } // end for cycles in calculation
        //------------- Done with calculation ------------------
        MATAR_FENCE();
        
        auto time_2 = std::chrono::high_resolution_clock::now();
        
        auto calc_time = std::chrono::duration_cast
                           <std::chrono::nanoseconds>(time_2 - time_1).count();
        
        
        // -------------------------------
        //    Print final state to a file
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



KOKKOS_INLINE_FUNCTION
int get_corners_in_cell(int cell_gid, int corner_lid){
    // corner_lid is 0 to 1
    return (2*cell_gid + corner_lid);
}

KOKKOS_INLINE_FUNCTION
int get_corners_in_node(int node_gid, int corner_lid){
    // corner_lid is 0 to 1
    return (2*node_gid - 1 + corner_lid);
}

// Function to plot density vs position using ASCII art
void plot_density_vs_position(const CArrayDual<double>& cell_coords, 
                            const CArrayDual<double>& cell_den,
                            const int num_cells,
                            const char* title) {
    // Clear screen and move cursor to top
    printf("\033[2J\033[H");
    
    const int PLOT_WIDTH = 120;  // Width of the plot in characters
    const int PLOT_HEIGHT = 20; // Height of the plot in characters
    const char PLOT_CHAR = '*'; // Character used for plotting
    
    // Find min and max values for scaling
    double min_x = cell_coords.host(0);
    double max_x = cell_coords.host(num_cells-1);
    double min_den = cell_den.host(0);
    double max_den = cell_den.host(0);
    
    for(int i = 0; i < num_cells; i++) {
        min_den = std::min(min_den, cell_den.host(i));
        max_den = std::max(max_den, cell_den.host(i));
    }
    
    // Add some padding to the density range
    double den_range = max_den - min_den;
    min_den -= 0.1 * den_range;
    max_den += 0.1 * den_range;
    
    // Create the plot
    printf("\n%s\n", title);
    printf("Density vs Position\n");
    printf("Density range: %.3f to %.3f\n", min_den, max_den);
    printf("Position range: %.1f to %.1f\n\n", min_x, max_x);
    
    // Plot the data
    for(int row = PLOT_HEIGHT-1; row >= 0; row--) {
        double den_value = min_den + (max_den - min_den) * row / (PLOT_HEIGHT-1);
        printf("%8.3f |", den_value);
        
        for(int col = 0; col < PLOT_WIDTH; col++) {
            double x_value = min_x + (max_x - min_x) * col / (PLOT_WIDTH-1);
            bool point_plotted = false;
            
            // Find the closest cell to this x position
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
    
    // Flush stdout to ensure immediate display
    fflush(stdout);
}



