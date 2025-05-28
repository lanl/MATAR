#include <stdio.h>
#include <iostream>
#include <chrono>
#include "matar.h"

// Required for MATAR data structures
using namespace mtr; 

const int    width  = 1000;
const int    height = 1000;
const double temp_tolerance = 0.01;
const int    max_iterations = 10000;



void initialize(CArrayDual<double>& temperature_previous);
void track_progress(int iteration, CArrayDual<double>& temperature);


// main
int main(int argc, char* argv[])
{    
    MATAR_INITIALIZE(argc, argv);
    { // MATAR scope
    
    CArrayDual<double> temperature(height + 2, width + 2);
    CArrayDual<double> temperature_previous(height + 2, width + 2);

    // 
    initialize(temperature_previous);

    // Start measuring time
    auto begin = std::chrono::high_resolution_clock::now();

    int iteration = 0;
    double worst_dt = 100;
    double max_value;

    while (worst_dt > temp_tolerance) {
        
        FOR_ALL(i, 1, height,
                j, 1, width, {
            temperature(i, j) = 0.25 * (  temperature_previous(i + 1, j)
                                        + temperature_previous(i - 1, j)
                                        + temperature_previous(i, j + 1)
                                        + temperature_previous(i, j - 1));
        });
        
        // calculate max difference between temperature and temperature_previous
        double local_max_value;


        FOR_REDUCE_MAX(i, 1, height,
                       j, 1, width,
                       loc_max_value, {
            double value = fabs(temperature(i, j) - temperature_previous(i, j));
            if (value > loc_max_value) {
                loc_max_value = value;
            }
        }, max_value);

        worst_dt = max_value;

        // update temperature_previous
        FOR_ALL(i, 1, height,
                j, 1, width, {
            temperature_previous(i, j) = temperature(i, j);
        });

        // track progress
        if (iteration % 100 == 0) {
            track_progress(iteration, temperature);
        }

        iteration++;    
    }


    // Stop measuring time and calculate the elapsed time
    auto end     = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf("Total time was %f seconds.\n", elapsed.count() * 1e-9);
    printf("\nMax error at iteration %d was %f\n", iteration - 1, worst_dt);

    }
    MATAR_FINALIZE();

    return 0;
}


void initialize(CArray<double>& temperature_previous)
{
    int i = 0;
    int j = 0;

    temperature_previous.set_values(0.0);

    FOR_ALL(i, 0, height + 1,{
        temperature_previous(i, 0) = 0.0; // left boundary
        temperature_previous(i, width + 1) = (100.0 / height) * i; // right boundary
    });

    FOR_ALL(j, 0, width + 1,{
        temperature_previous(0, j) = 0.0; // top boundary
        temperature_previous(height + 1, j) = (100.0 / width) * j; // bottom boundary
    });
}

void track_progress(int iteration, CArray<double>& temperature)
{
    int i = 0;
    temperature.update_host();

    printf("---------- Iteration number: %d ----------\n", iteration);
    for (i = height - 5; i <= height; i++) {
        printf("[%d,%d]: %5.2f  ", i, i, temperature.host(i, i));
    }
    printf("\n");
}