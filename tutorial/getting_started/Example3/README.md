# Health Data Analysis: An Introduction to Parallel Computing

This example demonstrates the power of **parallel computing** and **data-oriented programming** through the analysis of synthetic health data. By leveraging the MATAR library (built on top of Kokkos), we efficiently process data for millions of patients, showcasing how modern C++ and parallel abstractions can accelerate scientific computing.

## Problem Overview
We simulate health data for a large number of patients, each with several health features (e.g., blood pressure, cholesterol, etc.). The program computes:
- The **mean** of each feature
- The **variance** of each feature
- The **correlation matrix** between features

These are fundamental operations in statistics and data science, and are computationally intensive for large datasets—making them ideal for parallelization.

## Mathematical Background

### 1. Mean
For each feature \( j \), the mean is:
\[
\mu_j = \frac{1}{N} \sum_{i=1}^N x_{j,i}
\]
where \( x_{j,i} \) is the value of feature \( j \) for patient \( i \), and \( N \) is the number of patients.

### 2. Variance
For each feature \( j \), the (sample) variance is:
\[
\sigma_j^2 = \frac{1}{N-1} \sum_{i=1}^N (x_{j,i} - \mu_j)^2
\]

### 3. Correlation
The correlation between features \( j_1 \) and \( j_2 \) is:
\[
\rho_{j_1, j_2} = \frac{\sum_{i=1}^N (x_{j_1,i} - \mu_{j_1})(x_{j_2,i} - \mu_{j_2})}{(N-1)\sigma_{j_1}\sigma_{j_2}}
\]
This measures the linear relationship between two features.

## Parallel Programming Concepts

### Data-Oriented Programming
The code uses **data-oriented programming** by organizing data in contiguous arrays (using `CArrayDual` from MATAR), which enables efficient memory access and parallelization.

### Parallel Loops and Reductions
- **FOR_ALL**: Abstracts parallel loops over array indices, allowing the same code to run efficiently on CPUs or GPUs.
- **FOR_REDUCE_SUM**: Performs parallel reductions (e.g., summing values) across large datasets.

These macros are mapped to Kokkos parallel constructs, so the code is portable and scalable.

### Synchronization
- **MATAR_FENCE()**: Ensures all parallel operations are complete before proceeding, providing correct results and timing.
- **MATAR_INITIALIZE / MATAR_FINALIZE**: Set up and tear down the parallel execution environment.

## Code Structure
- **Data Generation**: Randomly generates health data in parallel.
- **Mean/Variance/Correlation**: Each statistic is computed using parallel loops and reductions.
- **Timers**: Measure the performance of each stage.

## Why Parallelism Matters
With 100 million patients and 10 features, a naive serial implementation would be extremely slow. By parallelizing the computation, we:
- Dramatically reduce runtime
- Make use of modern multi-core CPUs and GPUs
- Enable analysis of much larger datasets

## How to Run
1. **Build the code** (see CMakeLists.txt for dependencies; requires Kokkos and MATAR).
2. **Run the executable**:
   ```sh
   ./health_data
   ```
3. **Observe the output**: The program prints the mean, variance, and a section of the correlation matrix, along with timing information for each stage.

## Educational Takeaways
- **Parallel computing** is essential for large-scale data analysis.
- **Data-oriented design** and **parallel abstractions** (like those in MATAR/Kokkos) make it easier to write high-performance, portable code.
- Understanding the mathematics (mean, variance, correlation) is crucial for interpreting results and optimizing algorithms.

---

For more details, see the source code in `health_data.cpp` and explore other examples in the MATAR tutorial series.
