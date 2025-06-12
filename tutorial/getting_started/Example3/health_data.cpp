#include <chrono>
#include <iostream>
#include <random>


#include <matar.h>


constexpr int NumPatients = 1000000;  // Number of patients
constexpr int NumFeatures = 10;       // Number of health features

// Simple timer class for measuring execution time
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::string function_name;

public:
    // Constructor starts the timer
    Timer(const std::string& name) : function_name(name) {
        start_time = std::chrono::high_resolution_clock::now();
        std::cout << "Starting " << function_name << "...\n";
    }

    // Destructor automatically stops the timer and prints elapsed time
    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << function_name << " completed in " << duration.count() << " ms\n";
    }
};

// Function to generate synthetic health data
void generate_data(CArrayDual<double>& data) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(1234);
    
    FOR_ALL(i, 0, NumPatients,
            j, 0, NumFeatures, {
        auto gen = rand_pool.get_state();
        data(i,j) = gen.drand(50.0, 150.0);
        rand_pool.free_state(gen);
    });
}

// Compute mean of each feature
void compute_mean(const CArrayDual<double>& data, CArrayDual<double>& mean) {
    
    for (int j = 0; j < NumFeatures; j++) {
        
        double local_sum = 0.0;
        double global_sum = 0.0;

        FOR_REDUCE_SUM(i, 0, NumPatients,
                       j, 0, NumFeatures,
                       local_sum, {
            local_sum += data(i, j);
        }, global_sum);

        mean[j] = global_sum / NumPatients;
    }
}

// Compute variance of each feature
void compute_variance(const CArrayDual<double>& data, const CArrayDual<double>& mean, CArrayDual<double>& variance) {

    for (int j = 0; j < NumFeatures; j++) {
        double sum_sq = 0.0;

        FOR_REDUCE_SUM(i, 0, NumPatients,
                       j, 0, NumFeatures,
                       sum_sq, {
            sum_sq += std::pow(data(i, j) - mean[j], 2);
        }, sum_sq);

        variance[j] = sum_sq / (NumPatients - 1);  // Sample variance
    }
}

// Compute correlation matrix
void compute_correlation(const CArrayDual<double>& data, const CArrayDual<double>& mean, const CArrayDual<double>& variance, std::vector<std::vector<double>>& correlation) {

    for (int j1 = 0; j1 < NumFeatures; j1++) {
        for (int j2 = 0; j2 < NumFeatures; j2++) {
            if (j1 == j2) {
                correlation[j1][j2] = 1.0;  // Correlation with itself is always 1
                continue;
            }
            double sum = 0.0;

            for (int i = 0; i < NumPatients; i++) {
                sum += (data[i][j1] - mean[j1]) * (data[i][j2] - mean[j2]);
            }
            correlation[j1][j2] = sum / ((NumPatients - 1) * std::sqrt(variance[j1]) * std::sqrt(variance[j2]));
        }
    }
}

// Main function
int main() {
    std::cout << "Analyzing health data for " << NumPatients << " patients with " << NumFeatures << " features\n";
    
    {
        // Create timer for total runtime
        Timer timer("Total Runtime");

        // Create arrays for data, mean, variance, and correlation
        CArrayDual<double> data(NumPatients, NumFeatures);
        CArrayDual<double> mean(NumFeatures);
        CArrayDual<double> variance(NumFeatures);
        CArrayDual<double> correlation(NumFeatures, NumFeatures);

        // Set all values to 0
        data.set_values(0.0);
        mean.set_values(0.0);
        variance.set_values(0.0);
        correlation.set_values(0.0);
        
        {
            Timer timer("Generate Data");
            generate_data(data);
        }
        data.update_host();
        MATAT_FENCE();
        
        {
            Timer timer("Compute Mean");
            compute_mean(data, mean);
        }
        mean.update_host();
        MATAT_FENCE();
        
        {
            Timer timer("Compute Variance");
            compute_variance(data, mean, variance);
        }
        variance.update_host();
        MATAT_FENCE();

        {
            Timer timer("Compute Correlation");
            compute_correlation(data, mean, variance, correlation);
        }
        correlation.update_host();
        MATAT_FENCE();  

        std::cout << "\nHealth Feature Summary:\n";
        for (int j = 0; j < NumFeatures; j++) {
            std::cout << "Feature " << j << " - Mean: " << mean.host(j) << ", Variance: " << variance.host(j) << "\n";
        }

        std::cout << "\nCorrelation Matrix (first 3x3 section for brevity):\n";
        for (int j1 = 0; j1 < 3; j1++) {
            for (int j2 = 0; j2 < 3; j2++) {
                std::cout << correlation.host(j1, j2) << " ";
            }
            std::cout << "\n";
        }
    }

    return 0;
}