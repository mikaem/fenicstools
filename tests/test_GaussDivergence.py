import nose

from dolfin import VectorFunctionSpace, UnitSquareMesh, UnitCubeMesh,\
    interpolate, Expression

from fenicstools import *


def test_GaussDivergence():
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = interpolate(Expression(('x[0]', 'x[1]')), V)
    divu = gauss_divergence(u)
    DIVU = divu.vector().array()
    point_0 = all(abs(DIVU - 2.) < 1E-13)
    nose.tools.assert_equal(point_0, True)

    mesh = UnitCubeMesh(4, 4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = interpolate(Expression(('2*x[0]', '2*x[1]', '2*x[2]')), V)
    divu = gauss_divergence(u)
    DIVU = divu.vector().array()
    point_0 = all(abs(DIVU - 6.) < 1E-13)
    nose.tools.assert_equal(point_0, True)
