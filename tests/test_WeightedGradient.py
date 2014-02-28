import nose

from dolfin import FunctionSpace, UnitSquareMesh, interpolate, Function, Expression
from fenicstools import *

def test_WeightedGradient():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, 'CG', 1)
    u = interpolate(Expression("x[0]+2*x[1]"), V)
    wx = weighted_gradient_matrix(mesh, 0)
    dux = Function(V)
    dux.vector()[:] = wx * u.vector()
    wy = weighted_gradient_matrix(mesh, 1)
    duy = Function(V)
    duy.vector()[:] = wy * u.vector()
    nose.tools.assert_almost_equal(dux.vector().min(), 1.)
    nose.tools.assert_almost_equal(duy.vector().min(), 2.)
    
if __name__ == '__main__':
    nose.run(defaultTest=__name__)