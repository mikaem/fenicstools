#!/usr/bin/env py.test

from dolfin import VectorFunctionSpace, UnitSquareMesh, UnitCubeMesh,\
                   interpolate, Expression, MPI, mpi_comm_world
from fenicstools import gauss_divergence
import pytest

@pytest.fixture(scope="module", params=range(2))
def mesh(request):
    mesh = [UnitSquareMesh(4, 4), UnitCubeMesh(4, 4, 4)]
    return mesh[request.param]


def test_GaussDivergence(mesh):
    dim = mesh.topology().dim()
    expr = ["%s*x[%s]" % (dim,i) for i in range(dim)]
    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = interpolate(Expression(tuple(expr)), V)
    divu = gauss_divergence(u)
    DIVU = divu.vector().array()
    point_0 = all(abs(DIVU - dim*dim) < 1E-13)
    if MPI.rank(mpi_comm_world()) == 0:
        assert point_0
