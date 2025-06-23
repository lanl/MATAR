#include <stdio.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include "matar.h"

// Required for MATAR data structures
using namespace mtr; 

const int    width  = 300; // width of the grid not including boundaries
const int    height = 300; // height of the grid not including boundaries
const int    domain_width = width + 2; // width of the grid including boundaries
const int    domain_height = height + 2; // height of the grid including boundaries
const double temp_tolerance = 0.0005;
const int    max_iterations = 100000;

const double max_temp = 1000;

// Parameters for visualization
const int    vis_width = 60;    // Width of visualization grid
const int    vis_height = 20;   // Height of visualization grid
const bool   use_colors = true; // Set to false if terminal doesn't support colors

void initialize(CArrayDual<double>& temperature_previous);
void print_heatmap(CArrayDual<double>& temperature);
const char* temp_to_color(double temp, double max_temp);
char temp_to_char(double temp, double max_temp);

// main
int main()
{

    MATAR_INITIALIZE();
    { // MATAR scope

    int    i, j;
    int    iteration = 1;
    double worst_dt  = 100;

    // Start measuring time
    auto begin = std::chrono::high_resolution_clock::now();

    // Create the data structures to hold the temperature values
    CArrayDual<double> temperature(domain_height, domain_width);
    CArrayDual<double> temperature_previous(domain_height, domain_width);

    // initialize temperature profile
    initialize(temperature_previous);

    while (worst_dt > temp_tolerance) {
        // finite difference
        FOR_ALL(i, 1, height + 1,
                j, 1, width + 1, {
            temperature(i, j) = 0.25 * (  temperature_previous(i + 1, j)
                                        + temperature_previous(i - 1, j)
                                        + temperature_previous(i, j + 1)
                                        + temperature_previous(i, j - 1));
        });

        // calculate max difference between temperature and temperature_previous
        double local_max_value = 0.0;
        double max_value = 0.0;

        FOR_REDUCE_MAX(i, 1, height + 1,
                       j, 1, width + 1,
                       local_max_value, { // local_max_value is the value local to each thread
            
            double value = fabs(temperature(i, j) - temperature_previous(i, j));
            
            if (value > local_max_value) {
                local_max_value = value;
            }
            // update temperature_previous, not including boundaries
            temperature_previous(i, j) = temperature(i, j);
        }, max_value); // max_value is the maximum value of local_max_value across all threads

        // track progress
        if (iteration % 1000 == 0) {
            printf("---------- Iteration number: %d ----------\n", iteration);
            printf("\n");

            // Print the heatmap visualization for better understanding
            print_heatmap(temperature);
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

void initialize(CArrayDual<double>& temperature_previous)
{

    temperature_previous.set_values(0.0);


    FOR_ALL(i, 0, height + 1,{
        temperature_previous(i, 0) = 0.0; // left boundary
        temperature_previous(i, width + 1) = (1000.0 / height) * i; // right boundary
    });

    FOR_ALL(j, 0, width + 1,{
        temperature_previous(0, j) = 0.0; // top boundary
        temperature_previous(height + 1, j) = (1000.0 / width) * j; // bottom boundary
    });

    temperature_previous.update_host();
}


void print_heatmap(CArrayDual<double>& temperature)
{
    // Find the maximum temperature for scaling
    double max_temp = 1000;
    
    printf("\nTemperature Distribution (max = %.2f):\n", max_temp);
    printf("┌");
    for (int j = 0; j < vis_width; j++) printf("─");
    printf("┐\n");
    
    // Sample the grid to fit the visualization size
    for (int i = 0; i < vis_height; i++) {
        printf("│");
        for (int j = 0; j < vis_width; j++) {
            // Map visualization coordinates to actual grid coordinates
            int grid_i = 1 + (i * height / vis_height);
            int grid_j = 1 + (j * width / vis_width);
            
            double temp = temperature(grid_i,grid_j);
            
            if (use_colors) {
                printf("%s%c\033[0m", temp_to_color(temp, max_temp), temp_to_char(temp, max_temp));
            } else {
                printf("%c", temp_to_char(temp, max_temp));
            }
        }
        printf("│\n");
    }
    
    printf("└");
    for (int j = 0; j < vis_width; j++) printf("─");
    printf("┘\n");
    
    printf("Legend: . (cold) → * → o → O → # (hot)\n\n");
}

// ANSI color codes for terminal output
const char* temp_to_color(double temp, double max_temp) {
    double normalized = temp / max_temp;
    
    // Create a more gradual blue-to-red transition
    if (normalized < 0.1) return "\033[38;5;21m";  // Deep Blue
    if (normalized < 0.2) return "\033[38;5;27m";  // Medium Blue
    if (normalized < 0.3) return "\033[38;5;39m";  // Light Blue
    if (normalized < 0.4) return "\033[38;5;45m";  // Cyan-Blue
    if (normalized < 0.5) return "\033[38;5;51m";  // Cyan
    if (normalized < 0.6) return "\033[38;5;50m";  // Cyan-Green
    if (normalized < 0.7) return "\033[38;5;226m"; // Yellow
    if (normalized < 0.8) return "\033[38;5;214m"; // Orange
    if (normalized < 0.9) return "\033[38;5;208m"; // Dark Orange
    return "\033[38;5;196m";                       // Bright Red
}

// ASCII character for temperature intensity
char temp_to_char(double temp, double max_temp) {
    double normalized = temp / max_temp;
    
    // Match character intensity to the color gradient
    if (normalized < 0.1) return '*';
    if (normalized < 0.3) return '*';
    if (normalized < 0.5) return 'o';
    if (normalized < 0.7) return 'O';
    if (normalized < 0.9) return '@';
    return '#';
}