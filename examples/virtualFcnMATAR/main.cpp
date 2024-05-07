/**********************************************************************************************
 © 2020. Triad National Security, LLC. All rights reserved.
 This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
 National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
 Department of Energy/National Nuclear Security Administration. All rights in the program are
 reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
 Security Administration. The Government is granted for itself and others acting on its behalf a
 nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
 derivative works, distribute copies to the public, perform publicly and display publicly, and
 to permit others to do so.
 This program is open source under the BSD-3 License.
 Redistribution and use in source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:
 1.  Redistributions of source code must retain the above copyright notice, this list of
 conditions and the following disclaimer.
 2.  Redistributions in binary form must reproduce the above copyright notice, this list of
 conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.
 3.  Neither the name of the copyright holder nor the names of its contributors may be used
 to endorse or promote products derived from this software without specific prior
 written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************************************/
#include <iostream>
#include <stdio.h>
#include "matar.h"
#include "classes.hpp"

/*
 * 07/26/2023, Caleb Yenusah, Tested on cuda/11.4.0, rocm/5.4.3
 *
 * */

using namespace mtr;

// Pointer wrapper, because kokkos does not like pointers as template args
struct ShapePtr
{
    Shape* shape;
};

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    { // kokkos scope
        const size_t num_shapes = 4;
        DCArrayKokkos<ShapePtr> shape_array(num_shapes);

        // Allocate memory on GPU for shapes. Even=Circle, Odd=Square
        for (size_t i = 0; i < num_shapes; i++) {
            if (i % 2 == 0) {
                shape_array.host(i).shape = (Circle*)Kokkos::kokkos_malloc(sizeof(Circle));
            }
            else {
                shape_array.host(i).shape = (Square*)Kokkos::kokkos_malloc(sizeof(Square));
            }
        }
        // Update device side of array of memory location on GPU
        shape_array.update_device();

        // Create shapes using `placement new`. Even=Circle, Odd=Square. Radius=i, Length=i.
        FOR_ALL(i, 0, num_shapes, {
            if (i % 2 == 0) {
                new ((Circle*)shape_array(i).shape) Circle(i);
            }
            else {
                new ((Square*)shape_array(i).shape) Square(i);
            }
        });
        Kokkos::fence();

        // Calculate Area
        DCArrayKokkos<double> area_array(num_shapes);
        FOR_ALL(i, 0, num_shapes, {
            area_array(i) = shape_array(i).shape->area();
        });
        Kokkos::fence();
        area_array.update_host();

        // Check result
        for (size_t i = 0; i < num_shapes; i++) {
            double area;
            if (i % 2 == 0) {
                area = atan(1) * 4 * i * i;
                if (area != area_array.host(i)) {
                    printf("Circle radius=%.3f, calc_area=%.3f, actual_area=%.3f\n", i, area_array.host(i), area);
                }
            }
            else {
                area = i * i;
                if (area != area_array.host(i)) {
                    printf("Square length=%.3f, calc_area=%.3f, actual_area=%.3f\n", i, area_array.host(i), area);
                }
            }

            if (area != area_array.host(i)) {
                throw std::runtime_error("calculated area NOT EQUAL actual area");
            }
        }

        // Destroy shapes
        FOR_ALL(i, 0, num_shapes, {
            shape_array(i).shape->~Shape();
        });
        Kokkos::fence();

        // Free GPU memory
        for (size_t i = 0; i < num_shapes; i++) {
            Kokkos::kokkos_free(shape_array.host(i).shape);
        }

        printf("COMPLETED SUCCESSFULLY!!!\n");
    } // end kokkos scope
    Kokkos::finalize();

    return 0;
}
