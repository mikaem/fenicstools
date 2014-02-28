import nose

from dolfin import FunctionSpace, UnitCubeMesh, UnitSquareMesh, interpolate, \
    Expression, mpi_comm_self, VectorFunctionSpace
from fenicstools import *
from numpy import array

def test_Probe_functionspace_2D():
    mesh = UnitSquareMesh(mpi_comm_self(), 4, 4)
    V = FunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression('x[0]'), V)
    u0.update()    
    x = array([0.5, 0.5])

    p = Probe(x, V)
    p(u0)
    p(u0)
    nose.tools.assert_almost_equal(p[0][0], 0.5)
    nose.tools.assert_almost_equal(p[1][0], 0.5)

def test_Probe_functionspace_3D():
    mesh = UnitCubeMesh(mpi_comm_self(), 4, 4, 4)
    V = FunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression('x[0]'), V)
    u0.update()    
    x = array([0.25, 0.5, 0.5])
    
    p = Probe(x, V)
    p(u0)
    p(u0)
    nose.tools.assert_almost_equal(p[0][0], 0.25)
    nose.tools.assert_almost_equal(p[1][0], 0.25)

def test_Probe_vectorfunctionspace_2D():
    mesh = UnitSquareMesh(mpi_comm_self(), 4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression(('x[0]', 'x[1]')), V)
    u0.update()    
    x = array([0.5, 0.75])
    
    p = Probe(x, V)
    p(u0)
    p(u0)
    nose.tools.assert_almost_equal(p[0][0], 0.5)
    nose.tools.assert_almost_equal(p[1][1], 0.75)

def test_Probe_vectorfunctionspace_3D():
    mesh = UnitCubeMesh(mpi_comm_self(), 4, 4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), V)
    u0.update()    
    x = array([0.25, 0.5, 0.75])
    
    p = Probe(x, V)
    p(u0)
    p(u0)
    nose.tools.assert_almost_equal(p[0][0], 0.25)
    nose.tools.assert_almost_equal(p[1][1], 0.50)
    nose.tools.assert_almost_equal(p[1][2], 0.75)
    
if __name__ == '__main__':
    test_Probe_functionspace_3D()
    
