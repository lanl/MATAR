#include <stdio.h>
#include <matar.h>

// main
int main()
{
    // create array
    int size = 16;
    CArray <int> arr(size);
    
    FOR_ALL (i, 0, 16, {  // initialize with 1's
        arr(i) = 1;
    });

    FOR_ALL (i, 2, 16, {
        arr(i) = arr(i-1) + arr(i-2);  //add previous 2 indexes
    });

    int x;
    for(x = 0; x < size; x++) {
        printf("%d ", arr(x));  // print our array
    }
    printf("\n");
}

    
