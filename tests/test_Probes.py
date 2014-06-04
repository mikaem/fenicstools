import nose

from dolfin import FunctionSpace, UnitCubeMesh, UnitSquareMesh, interpolate, \
    Expression, MPI, mpi_comm_world, VectorFunctionSpace
from fenicstools import *
from numpy import array, load

def test_Probes_functionspace_2D():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression('x[0]'), V)
    x = array([[0.5, 0.5], [0.4, 0.4], [0.3, 0.3]])

    p = Probes(x.flatten(), V)
    # Probe twice
    p(u0)
    p(u0)
    
    # Check both snapshots
    p0 = p.array(N=0)
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0], 0.5)
        nose.tools.assert_almost_equal(p0[1], 0.4)
        nose.tools.assert_almost_equal(p0[2], 0.3)
    p0 = p.array(N=1)
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0], 0.5)
        nose.tools.assert_almost_equal(p0[1], 0.4)
        nose.tools.assert_almost_equal(p0[2], 0.3)
        
def test_Probes_functionspace_3D():
    mesh = UnitCubeMesh(4, 4, 4)
    V = FunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression('x[0]'), V)
    x = array([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]])
    
    p = Probes(x.flatten(), V)
    # Probe twice
    p(u0)
    p(u0)
    
    # Check both snapshots
    p0 = p.array(N=0)
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0], 0.5)
        nose.tools.assert_almost_equal(p0[1], 0.4)
        nose.tools.assert_almost_equal(p0[2], 0.3)
    p0 = p.array(N=1)
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0], 0.5)
        nose.tools.assert_almost_equal(p0[1], 0.4)
        nose.tools.assert_almost_equal(p0[2], 0.3)
        
    p0 = p.array(filename='dump')
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0, 0], 0.5)
        nose.tools.assert_almost_equal(p0[1, 1], 0.4)
        nose.tools.assert_almost_equal(p0[2, 1], 0.3)
        
        f = open('dump_all.probes', 'r')
        p1 = load(f)
        nose.tools.assert_almost_equal(p1[0, 0, 0], 0.5)
        nose.tools.assert_almost_equal(p1[1, 0, 1], 0.4)
        nose.tools.assert_almost_equal(p1[2, 0, 1], 0.3)
    
def test_Probes_vectorfunctionspace_2D():
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression(('x[0]', 'x[1]')), V)
    x = array([[0.5, 0.5], [0.4, 0.4], [0.3, 0.3]])
    
    p = Probes(x.flatten(), V)
    # Probe twice
    p(u0)
    p(u0)
    
    # Check both snapshots
    p0 = p.array(N=0)
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0, 0], 0.5)
        nose.tools.assert_almost_equal(p0[1, 1], 0.4)
        nose.tools.assert_almost_equal(p0[2, 1], 0.3)
    p0 = p.array(N=1)
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0, 0], 0.5)
        nose.tools.assert_almost_equal(p0[1, 0], 0.4)
        nose.tools.assert_almost_equal(p0[2, 1], 0.3)

    p0 = p.array(filename='dumpvector2D')
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0, 0, 0], 0.5)
        nose.tools.assert_almost_equal(p0[1, 1, 1], 0.4)
        nose.tools.assert_almost_equal(p0[2, 0, 1], 0.3)
        
        f = open('dumpvector2D_all.probes', 'r')
        p1 = load(f)
        nose.tools.assert_almost_equal(p1[0, 0, 0], 0.5)
        nose.tools.assert_almost_equal(p1[1, 1, 0], 0.4)
        nose.tools.assert_almost_equal(p1[2, 1, 1], 0.3)

def test_Probes_vectorfunctionspace_3D():
    mesh = UnitCubeMesh(4, 4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), V)
    x = array([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]])
    
    p = Probes(x.flatten(), V)
    # Probe twice
    p(u0)
    p(u0)
    
    # Check both snapshots
    p0 = p.array(N=0)
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0, 0], 0.5)
        nose.tools.assert_almost_equal(p0[1, 1], 0.4)
        nose.tools.assert_almost_equal(p0[2, 2], 0.3)
    p0 = p.array(N=1)
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0, 0], 0.5)
        nose.tools.assert_almost_equal(p0[1, 1], 0.4)
        nose.tools.assert_almost_equal(p0[2, 2], 0.3)
        
    p0 = p.array(filename='dumpvector3D')
    if MPI.rank(mpi_comm_world()) == 0:
        nose.tools.assert_almost_equal(p0[0, 0, 0], 0.5)
        nose.tools.assert_almost_equal(p0[1, 1, 0], 0.4)
        nose.tools.assert_almost_equal(p0[2, 1, 0], 0.3)
        
        f = open('dumpvector3D_all.probes', 'r')
        p1 = load(f)
        nose.tools.assert_almost_equal(p1[0, 0, 0], 0.5)
        nose.tools.assert_almost_equal(p1[1, 1, 0], 0.4)
        nose.tools.assert_almost_equal(p1[2, 1, 1], 0.3)

if __name__ == '__main__':
    nose.run(defaultTest=__name__)
    