#include <stdio.h>
#include <iostream>

#include "matar.h"


using array_type = FArray<int>;

int main()
{
  array_type A = array_type(2,2);
  array_type B = array_type(2,2);


  for (int j = 0; j < 2; ++j) {
  for (int i = 0; i < 2; ++i) {
    A(i,j) = 1;
    B(i,j) = 2;
  }
  }

  // using the overloaded operator+ 
  auto C = A + B;
  printf("Result should all be 3 \n");
  for (int j = 0; j < 2; ++j) {
  for (int i = 0; i < 2; ++i) {
      printf("%d\n", C(i,j));
  }
  }

  // also you can add multiple arrays
  C = A + A + B + B;
  printf("Result should all be 6 \n");
  for (int j = 0; j < 2; ++j) {
  for (int i = 0; i < 2; ++i) {
      printf("%d\n", C(i,j));
  }
  }

  // adding arrays of different dimensions is not allowed
  array_type D = array_type(2,2);
  array_type E = array_type(3,3);

  auto F = D + E; // error is thrown

  return 0;
}
