//
//  parent.h
//  
//
//  Created by nmorgan on 3/18/20.
//
#ifndef PARENT_H
#define PARENT_H
#include "child.hpp"


class parent_variables{
    
    public:
        //child_variables *child_var;
        int num_pnts;
        int type;
        // child variables
        double *child_p; // pressure
        //double *child_d; // density
        //double *child_sie; // specific internal energy
        //double *child_m; // mass
        //double *child_sspd; // sound speed
        //double *child_velgrad_matrix;  // size 9*num_mat_pts
        //double *child_stress_matrix;  // size 9*num_mat_pts

        // hypo strength variables
        double *hypo_fake1;
        //double *hypo_fake2;

    // ...
    
        // default constructor
        KOKKOS_FUNCTION
        parent_variables() {};

        // init constructor
        //parent_variables(int npnts, int mtype) {
        //    num_mat_pnts = npnts;
        //    mat_type     = mtype;
        //};
    
        // deconstructor
        KOKKOS_FUNCTION
        ~parent_variables(){};
    
};

class parent_models{
    
    public:
        child_models *child;
        //hypo_strength_models *hypo_strength;
    // ...
    
    
    // deconstructor
        KOKKOS_FUNCTION
        ~parent_models(){};
    
    
}; // end of parent


#endif
