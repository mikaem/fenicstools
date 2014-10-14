import nose

from dolfin import FunctionSpace, UnitSquareMesh, UnitCubeMesh, \
    interpolate, Function, Expression

from fenicstools import *

def test_WeightedGradient():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, 'CG', 1)
    u = interpolate(Expression("x[0]+2*x[1]"), V)
    wx = weighted_gradient_matrix(mesh, 0)
    du = Function(V)
    du.vector()[:] = wx * u.vector()
    nose.tools.assert_almost_equal(du.vector().min(), 1.)

    wy = weighted_gradient_matrix(mesh, 1)
    du.vector()[:] = wy * u.vector()
    nose.tools.assert_almost_equal(du.vector().min(), 2.)
    
    mesh = UnitCubeMesh(4, 4, 4)
    V = FunctionSpace(mesh, 'CG', 1)
    u = interpolate(Expression("x[0]+2*x[1]+3*x[2]"), V)
    du = Function(V)
    wx = weighted_gradient_matrix(mesh, (0, 1, 2))
    du.vector()[:] = wx[0] * u.vector()
    nose.tools.assert_almost_equal(du.vector().min(), 1.)
    du.vector()[:] = wx[1] * u.vector()
    nose.tools.assert_almost_equal(du.vector().min(), 2.)
    du.vector()[:] = wx[2] * u.vector()
    nose.tools.assert_almost_equal(du.vector().min(), 3.)
        
if __name__ == '__main__':
    nose.run(defaultTest=__name__)