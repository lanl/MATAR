#pragma once
#include "matar.h"

/* Shape */
class Shape {
public:
  KOKKOS_FUNCTION
  Shape();

  KOKKOS_FUNCTION
  virtual ~Shape();

  KOKKOS_FUNCTION
  virtual double area() = 0;
};


/* Circle */
class Circle : public Shape {
public:
  double radius;

  KOKKOS_FUNCTION
  Circle(double r);

  KOKKOS_FUNCTION
  ~Circle();

  KOKKOS_FUNCTION
  double area() override;
};


/* Square */
class Square : public Shape {
public:
  double length;

  KOKKOS_FUNCTION
  Square(double l);

  KOKKOS_FUNCTION
  ~Square();

  KOKKOS_FUNCTION
  double area() override;
};
