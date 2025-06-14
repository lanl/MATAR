program main
    ! This example demonstrates how to link Fortran with C++ using the ISO C Binding
    ! The program creates a 2D array in Fortran, passes it to C++ functions for processing,
    ! and receives the results back in Fortran

    ! ISO C Binding provides standardized ways to interoperate between Fortran and C/C++
    ! It ensures consistent data types and calling conventions between languages
    use iso_c_binding

    implicit none

    ! Define array dimensions as C-compatible integers
    ! Using c_int ensures the integer size matches C/C++ int type
    integer(kind=c_int), parameter :: nx = 4
    integer(kind=c_int), parameter :: ny = 4

    ! Declare a 2D array using C-compatible double precision
    ! c_double ensures the real number size matches C/C++ double type
    real(kind=c_double) :: array_2D(nx,ny)
    real(kind=c_double) :: sum_of_elements
    integer :: i, j, k, n

    ! initialize matar
    write(*,*)'initializing matar'
    call matar_initialize()

    ! Fill the 2D array with sequential numbers (1 to nx*ny)
    k = 0
    do j = 1, ny
        do i = 1, nx
          k = k+1
          array_2D(i,j) = k
        enddo
    enddo

    ! Call C++ function to square each element of the array
    ! The array is passed by reference (default in Fortran)
    ! The C++ function will modify the array in-place
    call square_array_elements(array_2D, nx, ny)

    ! Print the squared array elements to verify the C++ function worked
    print*, "printing squared array elements in fortran:"
    do j = 1, ny
        do i = 1, nx
          print*, array_2D(i,j)
        enddo
    enddo

    ! Call C++ function to sum all elements of the array
    ! The sum is returned through the sum_of_elements variable
    sum_of_elements = 0.0
    call sum_array_elements(array_2D, nx, ny, sum_of_elements)
    print*, "sum of elements in fortran = ", sum_of_elements

    ! Calculate and print the expected sum for verification
    ! The formula is the sum of squares from 1 to n: n(n+1)(2n+1)/6
    n = nx*ny;
    print*, "sum of elements should be ", (n*(n+1)*(2*n+1)) / 6

    ! Clean up matar resources
    write(*,*)'finalizing matar'
    call matar_finalize()

end program main
