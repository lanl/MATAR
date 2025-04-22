# MATAR Data Layout Example (C vs F)

This example demonstrates the differences between C-style (row-major) and F-style (column-major) data layouts in MATAR, and how they affect performance and memory access patterns.

## Overview

The example compares:
- CArray (C-style, row-major) vs FArray (F-style, column-major) layouts
- Performance implications of each layout
- Memory access patterns and cache utilization
- Best practices for choosing between layouts

## Data Layout Concepts

### C-Style (Row-Major) Layout
- Used by: C, C++, Python (NumPy default), Java, JavaScript
- Memory layout: Elements in the same row are stored contiguously
- Example for a 2D array:
  ```
  [a11, a12, a13, a21, a22, a23, a31, a32, a33]
  ```
- Best for: Row-wise operations, languages that use row-major by default

### F-Style (Column-Major) Layout
- Used by: Fortran, MATLAB, R, Julia
- Memory layout: Elements in the same column are stored contiguously
- Example for a 2D array:
  ```
  [a11, a21, a31, a12, a22, a32, a13, a23, a33]
  ```
- Best for: Column-wise operations, scientific computing applications

## Performance Considerations

### Cache Utilization
- C-Style: Better for row-wise access patterns
- F-Style: Better for column-wise access patterns

### Memory Access Patterns
- C-Style: Optimal when accessing elements row by row
- F-Style: Optimal when accessing elements column by column

### Language Integration
- When working with C/C++ code: Prefer CArray
- When working with Fortran code: Prefer FArray
- When using mixed language environments: Choose based on dominant access pattern

## Example Features

This example demonstrates:
1. Creating and initializing both CArray and FArray
2. Comparing performance of row-wise vs column-wise operations
3. Measuring memory access patterns
4. Analyzing cache utilization
5. Best practices for layout selection

## Building and Running

1. Compile with appropriate MATAR and Kokkos support:
   ```bash
   make
   ```

2. Run the example:
   ```bash
   ./data_layout
   ```

The program will:
1. Initialize MATAR
2. Create test arrays in both layouts
3. Perform operations with different access patterns
4. Measure and compare performance
5. Report findings and recommendations

## Performance Metrics

The example measures and reports:
- Memory access times for different patterns
- Cache hit/miss rates
- Overall operation performance
- Recommendations for layout selection

## Best Practices

1. Choose layout based on:
   - Primary access pattern (row-wise vs column-wise)
   - Language environment
   - Performance requirements
   - Integration needs

2. Consider data transformation costs when:
   - Converting between layouts
   - Working with mixed-language environments
   - Optimizing for specific hardware

3. Profile and measure:
   - Actual access patterns in your application
   - Cache utilization
   - Memory bandwidth usage
   - Overall performance impact
