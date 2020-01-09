#!/usr/bin/env py.test

import pytest
import numpy
from dolfin import *
from fenicstools import interpolate_nonmatching_mesh, interpolate_nonmatching_mesh_any

class Quadratic2D(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]*x[0] + x[1]*x[1] + 1.0

class Quadratic3D(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + 1.0

def test_functional2D():
    """Test integration of function interpolated in non-matching meshes"""

    f = Quadratic2D(degree=2)

    # Interpolate quadratic function on course mesh
    mesh0 = UnitSquareMesh(8, 8)
    V0 = FunctionSpace(mesh0, "Lagrange", 2)
    u0 = interpolate_nonmatching_mesh(f, V0)

    # Interpolate FE function on finer mesh
    mesh1 = UnitSquareMesh(31, 31)
    V1 = FunctionSpace(mesh1, "Lagrange", 2)
    u1 = interpolate_nonmatching_mesh(u0, V1)
    assert round(assemble(u0*dx) - assemble(u1*dx), 10) == 0

    mesh1 = UnitSquareMesh(30, 30)
    V1 = FunctionSpace(mesh1, "Lagrange", 2)
    u1 = interpolate_nonmatching_mesh(u0, V1)
    assert round(assemble(u0*dx) - assemble(u1*dx), 10) == 0

    f = Expression(("x[0]*x[0] + x[1]*x[1]",
                    "x[0]*x[0] + x[1]*x[1] + 1"), degree=2)
    V0 = FunctionSpace(mesh0, "Nedelec 1st kind H(curl)", 2)
    u0 = interpolate(f, V0)

    # Interpolate FE function on finer mesh
    V1 = FunctionSpace(mesh1, "Nedelec 1st kind H(curl)", 2)
    u1 = interpolate_nonmatching_mesh_any(u0, V1)
    assert round(assemble(dot(u0, u0)*dx) - assemble(dot(u1, u1)*dx), 4) == 0

    # Test with another expression
    f = Expression(("2*(x[0]*x[0] + x[1]*x[1])",
                    "2*(x[0]*x[0] + x[1]*x[1] + 1)"), degree=2)
    u0 = interpolate_nonmatching_mesh_any(f, V0)
    u1 = interpolate_nonmatching_mesh_any(u0, V1)
    assert round(assemble(dot(u0, u0)*dx) - assemble(dot(u1, u1)*dx), 4) == 0


def test_functional3D():
    """Test integration of function interpolated in non-matching meshes"""

    f = Quadratic3D(degree=2)

    # Interpolate quadratic function on course mesh
    mesh0 = UnitCubeMesh(8, 8, 8)
    V0 = FunctionSpace(mesh0, "Lagrange", 2)
    u0 = interpolate_nonmatching_mesh(f, V0)

    # Interpolate FE function on finer mesh
    mesh1 = UnitCubeMesh(21, 21, 21)
    V1 = FunctionSpace(mesh1, "Lagrange", 2)
    u1 = interpolate_nonmatching_mesh(u0, V1)
    assert round(assemble(u0*dx) - assemble(u1*dx), 10) == 0

    mesh1 = UnitCubeMesh(20, 20, 20)
    V1 = FunctionSpace(mesh1, "Lagrange", 2)
    u1 = interpolate_nonmatching_mesh(u0, V1)
    assert round(assemble(u0*dx) - assemble(u1*dx), 10) == 0

    f = Expression(("x[0]*x[0] + x[1]*x[1]",
                    "x[0]*x[0] + x[1]*x[1] + 1",
                    "x[0]*x[0] + x[1]*x[1] + 2"), degree=2)
    V0 = FunctionSpace(mesh0, "Nedelec 1st kind H(curl)", 2)
    u0 = interpolate(f, V0)

    # Interpolate FE function on finer mesh
    V1 = FunctionSpace(mesh1, "Nedelec 1st kind H(curl)", 2)
    u1 = interpolate_nonmatching_mesh_any(u0, V1)
    assert round(assemble(dot(u0, u0)*dx) - assemble(dot(u1, u1)*dx), 2) == 0
    
    # Test with another expression
    f = Expression(("2*(x[0]*x[0] + x[1]*x[1])",
                    "2*(x[0]*x[0] + x[1]*x[1] + 1)",
                    "2*(x[0]*x[0] + x[1]*x[1] + 2)"), degree=2)
    u0 = interpolate(f, V0)
    u1 = interpolate_nonmatching_mesh_any(u0, V1)
    assert round(assemble(dot(u0, u0)*dx) - assemble(dot(u1, u1)*dx), 1) == 0

if __name__ == '__main__':
    test_functional2D()
    #test_functional3D()
