#!/usr/bin/env py.test
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dolfin import VectorFunctionSpace, UnitSquareMesh, UnitCubeMesh,\
                   interpolate, Expression, MPI
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
    u = interpolate(Expression(tuple(expr), degree=1), V)
    divu = gauss_divergence(u)
    DIVU = divu.vector().get_local()
    point_0 = all(abs(DIVU - dim*dim) < 1E-13)
    if MPI.rank(MPI.comm_world) == 0:
        assert point_0

if __name__ == '__main__':
    mesh = UnitSquareMesh(4, 4)
    test_GaussDivergence(mesh)

