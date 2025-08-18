This example shows the MPI+MATAR implementation of the Laplace equation for steady-state temperature distribution using Jacobi iteration.
The test can be built with MPI+(serial,openMP,CUDA,HIP) kokkos backend.
The strong scaling results on multi-CPU and multi-GPU are detailed in "report.pdf".

To compile this code, a build script is provided in the MATAR/script folder.  
./build_matar.sh --kokkos_build_type= 

   &nbsp;&nbsp;&nbsp;&nbsp; serial_mpi
   
   &nbsp;&nbsp;&nbsp;&nbsp; openmp_mpi
   
   &nbsp;&nbsp;&nbsp;&nbsp; cuda_mpi
   
   &nbsp;&nbsp;&nbsp;&nbsp; hip_mpi

The user will need to install the appropriate third party libraries including MPI to compile the code.  Instructions are provide in the MATAR/script to install these libraries using Anaconda.

The Laplace solver application accepts two command line arguments `-height ${height} -width ${width}`. 
If the arguments are not provided default value of 1000 is used for both.

To run the Laplace solver app on the CPU using 4 cores:
`mpirun --bind-to core -n 4 ${EXEC} -height 2000 -width 2000`

To run the Laplace solver app with 4 cores, 2 cores per node, and 2 GPU per node:
`mpirun --bind-to core -n 4 --npernode 2 ${EXEC} -height 2000 -width 2000 --kokkos-num-devices=2`.
