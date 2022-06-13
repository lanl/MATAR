#include <stdio.h>
#include <chrono>
#include <math.h>
#include <matar.h>

// Calculation of Couette flow in a 2D box 

const int width = 1000;
const int height = 1000;

void initialize(CArray<double> &velocity_initial);

int main() {
    int i,j;
    int iteration = 0;

    auto velocity = CArray <double> (height+1, width+1);
    auto velocity_initial = CArray <double> (height+1, width+1);

    // Start the clock
    auto begin = std::chrono::high_resolution_clock::now();

    // initialize the velocity profile of the flow
    initialize(velocity_initial);

    while (iteration <= 1000) {
        // increment flow forward
        for (i = 0; i <= height-iteration; i++) {
            velocity(i,iteration) = 1;
        }
        iteration++;
    }

    // Stop counting time and calculate elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);

    printf("---------- Top Right Corner Diagonal Velocities ----------\n");
    for (i = 0; i <= height-995; i++) {
        for (j = width-5; j <= width; j++) {
            printf("[%d,%d]: %5.2f  ", i,j, velocity(i,j));
        }
    }
    printf("\n");

    printf("Total time was %f seconds.\n", elapsed.count() * 1e-9);
    
    return 0;
}

void initialize(CArray<double> &velocity_initial) {
    int i, j;

// setting the top and left boundary condition
    for (i = 0; i <= height; i++) {
        velocity_initial(i,0) = 1.0;
        velocity_initial(0,i) = 1.0;
    }
}

