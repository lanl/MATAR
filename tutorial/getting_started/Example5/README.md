# 1D Lagrangian Staggered Grid Hydro Solver

This code implements a 1D Lagrangian hydrodynamics solver using a staggered grid approach. The solver is designed to handle compressible fluid flows with shock waves and is implemented using MATAR+Kokkos for performance portability across CPUs and GPUs.

## Physics and Mathematical Formulation

The solver implements the Euler equations in Lagrangian form:

### Conservation of Mass
$$\frac{d\rho}{dt} + \rho \nabla \cdot \mathbf{v} = 0$$

### Conservation of Momentum
$$\rho \frac{d\mathbf{v}}{dt} = -\nabla P + \nabla \cdot \mathbf{\sigma}$$

### Conservation of Energy
$$\rho \frac{de}{dt} = -P \nabla \cdot \mathbf{v} + \mathbf{\sigma} : \nabla \mathbf{v}$$

where:
- $\rho$ is the density
- $\mathbf{v}$ is the velocity
- $P$ is the pressure
- $e$ is the specific internal energy
- $\mathbf{\sigma}$ is the artificial viscosity tensor

### Equation of State
The solver uses a gamma-law equation of state:
$$P = (\gamma - 1)\rho e$$

## Numerical Method

The solver uses a staggered grid approach where:
- Cell-centered quantities: density ($\rho$), pressure ($P$), specific internal energy ($e$)
- Node-centered quantities: velocity ($\mathbf{v}$), position ($\mathbf{x}$)

### Time Integration
The solver uses a second-order Runge-Kutta time integration scheme:
1. Predictor step: Calculate intermediate values
2. Corrector step: Update to final values

### Artificial Viscosity
The solver includes artificial viscosity to handle shock waves:
$$\sigma = \alpha \rho c_s |\Delta v| + \beta \rho (\Delta v)^2$$
where:
- $c_s$ is the sound speed
- $\Delta v$ is the velocity difference across a cell
- $\alpha$ and $\beta$ are coefficients

## Implementation Details

### Data Structures
- Cell variables: density, pressure, sound speed, specific internal energy, mass
- Node variables: velocity, position, mass
- Corner variables: force, mass, velocity

### Key Features
1. Lagrangian formulation (mesh moves with the fluid)
2. Staggered grid for improved stability
3. Artificial viscosity for shock capturing
4. Second-order Runge-Kutta time integration
5. Performance portable using MATAR+Kokkos

## Building and Running

### Prerequisites
- CMake 3.16 or higher
- C++17 compatible compiler
- Kokkos (automatically downloaded during build)

### Building
```bash
./build.sh -t <build_type>
```
where `<build_type>` can be:
- `serial`: Serial CPU execution
- `openmp`: OpenMP parallel execution
- `pthreads`: Pthreads parallel execution
- `cuda`: NVIDIA GPU execution
- `hip`: AMD GPU execution
- `all`: Build all available backends

Additional options:
- `-d`: Enable debug build
- `-v`: Enable vectorization verbose output

### Running
```bash
cd build_<backend>
./hydro
```

## Example Problems

The code includes two example problems:

1. Sod Shock Tube
   - Initial conditions: High pressure/density region on left, low pressure/density on right
   - Tests shock wave propagation and rarefaction waves

2. Sedov Blast Wave
   - Initial conditions: High energy point source in uniform medium
   - Tests spherical shock wave propagation

## Performance

The solver is designed for high performance using:
- Data-oriented programming principles
- Kokkos for performance portability
- Vectorization optimizations
- Efficient memory access patterns

## Visualization

The code includes a simple ASCII-based visualization of the density field that updates during the simulation. The visualization shows:
- Density vs. position
- Current state of the simulation
- Progress through the calculation
