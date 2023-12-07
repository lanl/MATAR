#include "classes.hpp"

/* Shape */
KOKKOS_FUNCTION
Shape::Shape() {}

KOKKOS_FUNCTION
Shape::~Shape() {}


/* Circle */
KOKKOS_FUNCTION
Circle::Circle(double r) : radius(r) {}

KOKKOS_FUNCTION
Circle::~Circle() {}

KOKKOS_FUNCTION
double Circle::area() {
  double result = atan(1)*4 * radius * radius;
  return result;
}

/* Square */
KOKKOS_FUNCTION
Square::Square(double l) : length(l) {}

KOKKOS_FUNCTION
Square::~Square() {}

KOKKOS_FUNCTION
double Square::area() {
  double result = length * length;
  return result;
}
