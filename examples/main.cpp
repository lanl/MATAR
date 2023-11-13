#include <stdio.h>

#include "matar.h"

using namespace mtr; // matar namespace

int main() {

    printf("Hello World\n");

    auto test = CArray <int> (5, 5); 

    test(3,3) = 10;

    printf("Succesfully made and used a CArray\n");

    return 0;
}
