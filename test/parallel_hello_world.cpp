#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "matar.h"
 
int main(int argc, char* argv[])
{
    int thread_id;
    // Beginning of parallel region
    #pragma omp parallel
    {
        thread_id = omp_get_thread_num();
        printf("Hello World... from thread = %d\n", thread_id);
    }
    // Ending of parallel region
}