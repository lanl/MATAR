/**********************************************************************************************
 � 2020. Triad National Security, LLC. All rights reserved.
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
//
// Using FOR_ALL_HOST, and mixing host and device parallel loops
//
// Two things this example shows:
//
//   1. How to write CPU-side parallel work with the _HOST macro family
//      (FOR_ALL_HOST, DO_ALL_HOST, RUN_HOST, FOR_REDUCE_*_HOST) -- including
//      the work a device kernel can never do, such as touching std::string,
//      std::vector, or a stream.
//
//   2. How to combine the two families: a FOR_ALL running on the device while
//      a FOR_ALL_HOST runs on the host, which fences belong between them, and
//      how a Dual type hands data from one side to the other.
//
// The _HOST macros differ from their device twins in exactly two ways:
//
//   * the loop runs in the host execution space (DefaultHostExecSpace, picked
//     by MATAR_HOST_BACKEND), never on the GPU; and
//   * the lambda captures by reference ([&] instead of KOKKOS_LAMBDA), so host
//     objects that are not device-copyable can be used directly, and no
//     separate _CLASS form is needed inside a member function.
//
// Everything else -- argument forms, the optional trailing kernel-name string,
// FOR_ vs DO_ bounds -- is identical to the device macros.
//
// Build and run. Overlap is only real when the two sides are different
// execution spaces, e.g. a CUDA device with an OpenMP host:
//
//   cmake -B build/overlap -DMATAR_ENABLE_KOKKOS=ON -DMATAR_BUILD_EXAMPLES=ON \
//         -DMATAR_DEVICE_BACKEND=cuda -DMATAR_HOST_BACKEND=openmp \
//         -DKokkos_ARCH_NATIVE=ON
//   cmake --build build/overlap -j
//   ./build/overlap/bin/host_device_overlap
//
// The example is correct in every configuration (serial, openmp, cuda, ...);
// with one backend driving both sides the host and device kernels simply take
// turns instead of overlapping.

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "matar.h"

using namespace mtr;  // matar namespace

// ---------------------------------------------------------------------------
// A host kernel inside a member function needs no _CLASS variant: [&] captures
// `this` on its own. FOR_ALL_HOST_CLASS exists as an alias for symmetry with
// the device macros, and expands to exactly the same thing.
// ---------------------------------------------------------------------------
struct HostField {
    CArrayHost<double> values;

    HostField(int n) : values(n) {}

    void fill_with_squares() {
        const int n = (int)values.dims(0);
        FOR_ALL_HOST(i, 0, n, {
            values(i) = (double)i * (double)i;  // `values` reached through a captured `this`
        }, "host_field_fill");
    }
};

// A device kernel must live where its lambda is instantiated once: a .cpp
// file, never a header shared by several translation units. (KOKKOS_LAMBDA
// becomes an nvcc extended lambda whose thunks are TU-local.)
void launch_device_work(CArrayDevice<double>& work, int n, int inner_iters) {
    FOR_ALL(i, 0, n, {
        // enough arithmetic that the kernel is still running when the launch
        // returns, so the host has something to overlap with
        double acc = (double)i;
        for (int k = 0; k < inner_iters; k++) {
            acc = acc * 1.0000001 + 0.5;
        }
        work(i) = acc;
    }, "device_work");
    // deliberately NOT fenced: on a GPU build this returns while the kernel is
    // still in flight, which is the whole point of the host/device split
}

double device_sum(CArrayDevice<double>& work, int n) {
    double total     = 0.0;
    double loc_total = 0.0;
    FOR_REDUCE_SUM(i, 0, n,
                   loc_total, {
        loc_total += work(i);
    }, total);
    // a reduction fences on its own: no MATAR_FENCE() needed before reading
    // `total` on the host
    return total;
}

void double_on_device(CArrayDual<double>& field, int n) {
    FOR_ALL(i, 0, n, {
        field(i) = 2.0 * field(i);  // no .host(): inside a device kernel the
                                    // plain accessor is the device copy
    }, "double_dual_field");
}

// main
int main(int argc, char* argv[]) {
    MATAR_INITIALIZE(argc, argv);
    {
        // ===================================================================
        // Part 1: the _HOST macros on their own
        //
        // Host types (CArrayHost, CMatrixHost, ...) always live in CPU memory,
        // so they are what a host kernel is allowed to touch.
        // ===================================================================
        printf("--- Part 1: FOR_ALL_HOST / DO_ALL_HOST basics ---\n");

        const int nx = 4;
        const int ny = 3;

        CArrayHost<double> a1(nx);
        CArrayHost<double> a2(nx, ny);

        // 1D. Same argument form as FOR_ALL: index, start, end (half open).
        FOR_ALL_HOST(i, 0, nx, {
            a1(i) = (double)i;
        });

        // 2D, with the optional trailing kernel name used for profiling
        FOR_ALL_HOST(i, 0, nx,
                     j, 0, ny, {
            a2(i, j) = 10.0 * (double)i + (double)j;
        }, "host_fill_a2");

        printf("a1(3)    = %4.1f  (expected  3.0)\n", a1(3));
        printf("a2(3, 2) = %4.1f  (expected 32.0)\n", a2(3, 2));

        // DO_ALL_HOST takes INCLUSIVE bounds and pairs with the 1-based Matrix
        // types, exactly as DO_ALL does on the device.
        CMatrixHost<double> m(nx, ny);
        DO_ALL_HOST(i, 1, nx,
                    j, 1, ny, {
            m(i, j) = 100.0 * (double)i + (double)j;
        });
        printf("m(%d, %d)   = %6.1f  (expected %6.1f)\n", nx, ny, m(nx, ny), 100.0 * nx + ny);

        // RUN_HOST executes its body once, in the host space
        int host_once = 0;
        RUN_HOST({
            host_once = 42;
        });
        printf("RUN_HOST wrote %d\n\n", host_once);

        // ===================================================================
        // Part 2: what the _HOST macros buy you -- capture by reference
        //
        // These bodies could not be written with FOR_ALL: a KOKKOS_LAMBDA
        // captures by value and must be copyable to the device, which rules
        // out std::string, std::vector, streams and file handles.
        // ===================================================================
        printf("--- Part 2: host-only objects inside a host kernel ---\n");

        const int n_rows = 6;
        std::vector<std::string> report(n_rows);
        const std::string prefix = "row-";

        // Each iteration writes its own slot, so there is no race. Anything
        // shared and mutable (a push_back, an accumulator, one stream) would
        // need a reduction or a lock, just as in any threaded code.
        FOR_ALL_HOST(i, 0, n_rows, {
            std::ostringstream os;
            os << prefix << i << " : " << a1(i % nx);
            report[i] = os.str();
        }, "build_report");

        std::cout << "  " << report[0] << "\n  " << report[n_rows - 1] << "\n";

        // host reductions mirror the device ones, including the thread-local
        // accumulator argument
        HostField field(16);
        field.fill_with_squares();  // FOR_ALL_HOST inside a member function

        double sum_sq     = 0.0;
        double loc_sum_sq = 0.0;
        FOR_REDUCE_SUM_HOST(i, 0, 16,
                            loc_sum_sq, {
            loc_sum_sq += field.values(i);
        }, sum_sq);

        double max_sq     = 0.0;
        double loc_max_sq = 0.0;
        FOR_REDUCE_MAX_HOST(i, 0, 16,
                            loc_max_sq, {
            if (field.values(i) > loc_max_sq) {
                loc_max_sq = field.values(i);
            }
        }, max_sq);

        printf("  sum of i*i for i in [0,16) = %6.1f  (expected 1240.0)\n", sum_sq);
        printf("  max of i*i for i in [0,16) = %6.1f  (expected  225.0)\n\n", max_sq);

        // ===================================================================
        // Part 3: a device FOR_ALL and a host FOR_ALL_HOST at the same time
        //
        // A device parallel_for is asynchronous: the launch returns while the
        // kernel runs. A host parallel_for blocks the calling thread but not
        // the GPU. So the pattern is: launch the device work, do the host work,
        // then fence the device before reading what it wrote.
        // ===================================================================
        printf("--- Part 3: overlapping host and device work ---\n");

        // the two sides run the same recurrence over the same range, so their
        // sums must agree to rounding -- a cheap check that neither kernel
        // lost work to the overlap
        const int n_dev       = 1 << 20;
        const int n_host      = n_dev;
        const int inner_iters = 400;

        CArrayDevice<double> work(n_dev, "device_work");
        CArrayHost<double> host_work(n_host);

        auto t0 = std::chrono::high_resolution_clock::now();

        // (a) launch on the device and do NOT fence
        launch_device_work(work, n_dev, inner_iters);

        auto t1 = std::chrono::high_resolution_clock::now();

        // (b) host work proceeds while the device kernel is in flight. It may
        //     only touch host-accessible data -- `host_work` here, never
        //     `work`.
        FOR_ALL_HOST(i, 0, n_host, {
            double acc = (double)i;
            for (int k = 0; k < inner_iters; k++) {
                acc = acc * 1.0000001 + 0.5;
            }
            host_work(i) = acc;
        }, "host_work");

        // A host parallel_for has completed by the time it returns, so this
        // fence is not required; it is spelled out to show that it waits on
        // the HOST space only and leaves the device kernel running.
        MATAR_FENCE_HOST();

        auto t2 = std::chrono::high_resolution_clock::now();

        // (c) now wait for the device side. MATAR_FENCE_DEVICE() fences only
        //     the device space; MATAR_FENCE() would fence both.
        MATAR_FENCE_DEVICE();

        auto t3 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> launch_ms = t1 - t0;
        std::chrono::duration<double, std::milli> host_ms   = t2 - t1;
        std::chrono::duration<double, std::milli> drain_ms  = t3 - t2;

        printf("  device launch returned after %8.3f ms (async on a GPU build)\n", launch_ms.count());
        printf("  host kernel took             %8.3f ms (ran alongside the device)\n", host_ms.count());
        printf("  device fence then waited     %8.3f ms (what was left of the kernel)\n", drain_ms.count());

        const double dev_total = device_sum(work, n_dev);
        double host_total      = 0.0;
        double loc_host_total  = 0.0;
        FOR_REDUCE_SUM_HOST(i, 0, n_host,
                            loc_host_total, {
            loc_host_total += host_work(i);
        }, host_total);

        printf("  device sum = %.6e   host sum = %.6e (same recurrence)\n\n", dev_total, host_total);

        // ===================================================================
        // Part 4: handing data between the two sides with a Dual type
        //
        // A *Device array is device memory: a host kernel cannot read it. To
        // work on the same field from both sides, declare it Dual and use the
        // .host() accessor in host kernels, the plain accessor in device
        // kernels, with update_device() / update_host() in between.
        // ===================================================================
        printf("--- Part 4: Dual type handoff ---\n");

        const int n = 8;
        CArrayDual<double> field_dual(n, "dual_field");

        // (a) fill on the host, in parallel, through .host()
        FOR_ALL_HOST(i, 0, n, {
            field_dual.host(i) = (double)i;
        }, "dual_host_fill");

        // (b) push the host copy to the device before the device reads it
        field_dual.update_device();

        // (c) device kernel doubles every entry
        double_on_device(field_dual, n);

        // (d) fence before pulling the result back: update_host() must not
        //     start while the kernel that produced the data is still running
        MATAR_FENCE();
        field_dual.update_host();

        // (e) and now the host side is current again
        printf("  after host fill + device doubling: ");
        for (int i = 0; i < n; i++) {
            printf("%.0f ", field_dual.host(i));
        }
        printf("\n  (expected 0 2 4 6 8 10 12 14)\n\n");

        // ===================================================================
        // The one rule that bites
        //
        // A host kernel may only touch host-accessible memory: a *Host type,
        // the .host() side of a Dual type, or plain std:: data. Reaching into
        // a *Device array from FOR_ALL_HOST compiles cleanly and then aborts
        // at run time on a GPU build with "attempt to access inaccessible
        // memory space":
        //
        //     CArrayDevice<double> d(n);
        //     FOR_ALL_HOST(i, 0, n, {
        //         d(i) = 1.0;             // WRONG: device memory, host kernel
        //     });
        //
        // Declare the field Dual and use .host(), as in Part 4 above.
        // ===================================================================
        printf("done\n");

    }  // end of kokkos scope

    MATAR_FINALIZE();

    return 0;
}  // end of main
