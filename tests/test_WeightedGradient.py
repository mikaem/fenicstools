#!/usr/bin/env py.test

import pytest
from dolfin import FunctionSpace, UnitSquareMesh, UnitCubeMesh, \
                   interpolate, Function, Expression
from fenicstools import *

@pytest.fixture(scope="module", params=range(2))
def mesh(request):
    mesh = [UnitSquareMesh(4, 4), UnitCubeMesh(4, 4, 4)]
    return mesh[request.param]


@pytest.fixture(scope="module")
def V(mesh):
    return FunctionSpace(mesh, 'CG', 1)


def test_WeightedGradient(V, mesh):
    dim = mesh.topology().dim()
    expr = "+".join(["%d*x[%d]" % (i+1,i) for i in range(dim)])
    u = interpolate(Expression(expr), V)
    du = Function(V)
    wx = weighted_gradient_matrix(mesh, tuple(range(dim)))
    for i in range(dim):
        du.vector()[:] = wx[i] * u.vector()
        assert round(du.vector().min() - (i+1), 7) == 0
