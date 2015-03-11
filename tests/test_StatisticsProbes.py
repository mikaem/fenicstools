import nose

from dolfin import FunctionSpace, UnitCubeMesh, UnitSquareMesh, interpolate, \
    Expression, VectorFunctionSpace, MPI, mpi_comm_world
from fenicstools import *
from numpy import array

def test_StatisticsProbes_segregated_2D():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression('x[0]'), V)
    v0 = interpolate(Expression('x[1]'), V)
    x = array([[0.5, 0.25], [0.4, 0.4], [0.3, 0.3]])
    probes = StatisticsProbes(x.flatten(), V, True)

    for i in range(5):
        probes(u0, v0)
        
    p = probes.array()
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p[0,0], 2.5)
        nose.tools.assert_almost_equal(p[0,4], 0.625)
        
def test_StatisticsProbes_segregated_3D():
    mesh = UnitCubeMesh(4, 4, 4)
    V = FunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression('x[0]'), V)
    v0 = interpolate(Expression('x[1]'), V)
    w0 = interpolate(Expression('x[2]'), V)
    x = array([[0.5, 0.25, 0.25], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]])
    probes = StatisticsProbes(x.flatten(), V, True)

    for i in range(5):
        probes(u0, v0, w0)
        
    p = probes.array()
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p[0,0], 2.5)
        nose.tools.assert_almost_equal(p[0,4], 0.3125)

def test_StatisticsProbes_vector_2D():
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression(('x[0]', 'x[1]')), V)
    x = array([[0.5, 0.25], [0.4, 0.4], [0.3, 0.3]])
    probes = StatisticsProbes(x.flatten(), V)

    for i in range(5):
        probes(u0)
        
    p = probes.array()
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p[0,0], 2.5)
        nose.tools.assert_almost_equal(p[0,4], 0.625)

def test_StatisticsProbes_vector_3D():
    mesh = UnitCubeMesh(4, 4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), V)
    x = array([[0.5, 0.25, 0.25], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]])
    probes = StatisticsProbes(x.flatten(), V)

    for i in range(5):
        probes(u0)
        
    p = probes.array()
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p[0,0], 2.5)
        nose.tools.assert_almost_equal(p[0,4], 0.3125)

if __name__ == '__main__':
    nose.run(defaultTest=__name__)