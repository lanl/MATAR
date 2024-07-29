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
#include "profile.h"
#include <Kokkos_Core.hpp>

//
// Declaration of Event class
//
Event::Event(const std::string& name)
{
    name_  = name;
    count_ = 0;
    // total_time_ = duration_t::duration::zero();
    clock_t::time_point t1 = clock_t::now();
    total_time_ = t1 - t1;
}

void Event::start()
{
    start_time_ = clock_t::now();
}

void Event::stop()
{
    count_ += 1;
    total_time_ += (clock_t::now() - start_time_);
}

double Event::get_time_in_seconds()
{
    return std::chrono::duration_cast
           <std::chrono::nanoseconds>
           (total_time_).count() * 1.0e-9;
}

int Event::get_count()
{
    return count_;
}

std::string& Event::get_name()
{
    return name_;
}

//
// Declaration of Profile class
//
Event Profile::total = Event("total");
Event Profile::fft_forward  = Event("fft_forward");
Event Profile::fft_backward = Event("fft_backward");

//
std::vector<Event*> Profile::events_;

Profile::Profile()
{
}

void Profile::start(Event& event)
{
    event.start();
}

void Profile::stop(Event& event)
{
    if (event.get_count() == 0) {
        events_.push_back(&event);
    }

    event.stop();
}

void Profile::start_barrier(Event& event)
{
    Kokkos::fence();
#ifdef HAVE_CUDA
    cudaDeviceSynchronize();
#elif HAVE_OPENMP
  #pragma omp barrier
#endif
    event.start();
}

void Profile::stop_barrier(Event& event)
{
    if (event.get_count() == 0) {
        events_.push_back(&event);
    }

    Kokkos::fence();
#ifdef HAVE_CUDA
    cudaDeviceSynchronize();
#elif HAVE_OPENMP
  #pragma omp barrier
#endif
    event.stop();
}

void Profile::print_one(Event& event)
{
    // printf("%s : %12.4E seconds\n", event.get_name().c_str(), event.get_time_in_seconds());
    printf("\n");
    printf("%s:\n", event.get_name().c_str());
    printf("  time: %12.4E seconds", event.get_time_in_seconds());
    printf("  count: %d", event.get_count());
    printf("  fraction: %12.4E%%", event.get_time_in_seconds() / total.get_time_in_seconds() * 100.0);
    printf("\n");
}

void Profile::print()
{
    for (Event* event : events_) {
        print_one(*event);
    }
}
