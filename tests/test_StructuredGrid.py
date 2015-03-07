#!/usr/bin/env py.test

import pytest
from dolfin import FunctionSpace, UnitCubeMesh, UnitSquareMesh, interpolate, \
                   Expression, set_log_level, MPI, mpi_comm_world, mpi_comm_self, \
                   VectorFunctionSpace
from fenicstools import *
from fixtures import *
#try:
import mpi4py
#except:
#    assert False

@pytest.fixture(scope="module")
def mesh():
    return UnitCubeMesh(16, 16, 16)

@pytest.fixture(scope="module")
def V(mesh):
    return FunctionSpace(mesh, 'CG', 1)

@pytest.fixture(scope="module")
def Vv(mesh):
    return VectorFunctionSpace(mesh, 'CG', 1)

@pytest.fixture(scope="module")
def W(V, Vv):
    return V * Vv

@pytest.fixture(scope="module")
def s0(V):
    s0 = interpolate(Expression('exp(-(pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2) + pow(x[2]-0.5, 2)))'), V)
    return s0

@pytest.fixture(scope="module")
def v0(Vv):
    return interpolate(Expression(('x[0]', '2*x[1]', '3*x[2]')), Vv)

@pytest.fixture(scope="module")
def w0(W):
    return interpolate(Expression(('x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]')), W)

@pytest.fixture(scope="module")
def x(V):
    x0 = interpolate(Expression('x[0]'), V)
    y0 = interpolate(Expression('x[1]'), V)
    z0 = interpolate(Expression('x[2]'), V)
    return [x0, y0, z0]


def test_StructuredGrid_Box(s0, V, dirpath):
    origin = [0.25, 0.25, 0.25]                 # origin of box
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # coordinate vectors (scaled in StructuredGrid)
    dL = [0.5, 0.5, 0.5]                        # extent of slice in both directions
    N  = [9, 9, 6]                              # number of points in each direction

    sl = StructuredGrid(V, N, origin, vectors, dL)
    sl(s0)     # probe once
    sl(s0)     # probe once more
    sl.tovtk(0, filename=dirpath+"dump_scalar.vtk")
    sl.toh5(0, 1, dirpath+'dump_scalar.h5', dtype='float64')
    sl2 = StructuredGrid(V, restart=dirpath+'dump_scalar.h5')

    assert sl.dL[0] == sl2.dL[0] and sl.dL[1] == sl2.dL[1] and sl.dL[2] == sl2.dL[2] 
    assert sl.arithmetic_mean() == sl2.arithmetic_mean()


def test_StructuredGrid_Slice(s0, V, dirpath):
    # 2D slice
    origin = [-0.5, -0.5, 0.5]            # origin of slice
    vectors = [[1, 0, 0], [0, 1, 0]]      # directional tangent directions (scaled in StructuredGrid)
    dL = [2., 2.]                         # extent of slice in both directions
    N  = [50, 50]                         # number of points in each direction

    sl = StructuredGrid(V, N, origin, vectors, dL)
    sl(s0)     # probe once
    sl(s0)     # probe once more
    sl.tovtk(0, filename=dirpath+"dump_scalar.vtk")
    sl.toh5(0, 1, dirpath+'dump_scalar.h5', dtype='float64')
    sl2 = StructuredGrid(V, restart=dirpath+'dump_scalar.h5')

    assert sl.dL[0] == sl2.dL[0] and sl.dL[1] == sl2.dL[1] 
    assert sl.arithmetic_mean() == sl2.arithmetic_mean()


# then vector
def test_StructuredGrid_Box_vector(v0, Vv, dirpath):
    origin = [0.25, 0.25, 0.25]                 # origin of box
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # coordinate vectors (scaled in StructuredGrid)
    dL = [0.5, 0.5, 0.5]                        # extent of slice in both directions
    N  = [9, 9, 6]                              # number of points in each direction

    sl = StructuredGrid(Vv, N, origin, vectors, dL)
    sl(v0)     # probe once
    sl(v0)     # probe once more
    sl.tovtk(0, filename=dirpath+"dump_vector.vtk")
    sl.toh5(0, 1, dirpath+'dump_vector.h5', dtype='float64')
    sl2 = StructuredGrid(Vv, restart=dirpath+'dump_vector.h5')

    assert sl.dL[0] == sl2.dL[0] and sl.dL[1] == sl2.dL[1] and sl.dL[2] == sl2.dL[2] 
    assert sl.arithmetic_mean() == sl2.arithmetic_mean()


def test_StructuredGrid_Slice_vector(v0, Vv, dirpath):
    # 2D slice
    origin = [-0.5, -0.5, 0.5]            # origin of slice
    vectors = [[1, 0, 0], [0, 1, 0]]      # directional tangent directions (scaled in StructuredGrid)
    dL = [2., 2.]                         # extent of slice in both directions
    N  = [50, 50]                         # number of points in each direction

    sl = StructuredGrid(Vv, N, origin, vectors, dL)
    sl(v0)     # probe once
    sl(v0)     # probe once more
    sl.tovtk(0, filename=dirpath+"dump_vector.vtk")
    sl.toh5(0, 1, dirpath+'dump_vector.h5', dtype='float64')
    sl2 = StructuredGrid(Vv, restart=dirpath+'dump_vector.h5')

    assert sl.dL[0] == sl2.dL[0] and sl.dL[1] == sl2.dL[1] 
    assert sl.arithmetic_mean() == sl2.arithmetic_mean()


# then vector
def test_StructuredGrid_Box_mixed(w0, W, dirpath):
    origin = [0.25, 0.25, 0.25]                 # origin of box
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # coordinate vectors (scaled in StructuredGrid)
    dL = [0.5, 0.5, 0.5]                        # extent of slice in both directions
    N  = [9, 9, 6]                              # number of points in each direction

    sl = StructuredGrid(W, N, origin, vectors, dL)
    sl(w0)     # probe once
    sl(w0)     # probe once more
    sl.toh5(0, 1, dirpath+'dump_mixed.h5', dtype='float64')
    sl2 = StructuredGrid(W, restart=dirpath+'dump_mixed.h5')

    assert sl.dL[0] == sl2.dL[0] and sl.dL[1] == sl2.dL[1] and sl.dL[2] == sl2.dL[2] 
    assert sl.arithmetic_mean() == sl2.arithmetic_mean()


def test_StructuredGrid_Slice_mixed(w0, W, dirpath):
    # 2D slice
    origin = [-0.5, -0.5, 0.5]            # origin of slice
    vectors = [[1, 0, 0], [0, 1, 0]]      # directional tangent directions (scaled in StructuredGrid)
    dL = [2., 2.]                         # extent of slice in both directions
    N  = [50, 50]                         # number of points in each direction

    sl = StructuredGrid(W, N, origin, vectors, dL)
    sl(w0)     # probe once
    sl(w0)     # probe once more
    sl.toh5(0, 1, dirpath+'dump_mixed.h5', dtype='float64')
    sl2 = StructuredGrid(W, restart=dirpath+'dump_mixed.h5')

    assert sl.dL[0] == sl2.dL[0] and sl.dL[1] == sl2.dL[1] 
    assert round(sl.arithmetic_mean() - sl2.arithmetic_mean(), 7) == 0


def test_StructuredGrid_Box_vector_statistics(v0, Vv, dirpath):
    origin = [0.25, 0.25, 0.25]                 # origin of box
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # coordinate vectors (scaled in StructuredGrid)
    dL = [0.5, 0.5, 0.5]                        # extent of slice in both directions
    N  = [9, 9, 6]                              # number of points in each direction

    sl = StructuredGrid(Vv, N, origin, vectors, dL, statistics=True)
    sl(v0)     # probe once
    sl(v0)     # probe once more
    sl.tovtk(0, filename=dirpath+"dump_stats.vtk")
    sl.toh5(0, 1, dirpath+'dump_stats.h5', dtype='float64')
    sl2 = StructuredGrid(Vv, restart=dirpath+'dump_stats.h5')

    assert sl.dL[0] == sl2.dL[0] and sl.dL[1] == sl2.dL[1] and sl.dL[2] == sl2.dL[2] 
    assert round(sl.arithmetic_mean() - sl2.arithmetic_mean(), 7) == 0
    assert sl.probes.number_of_evaluations() == sl2.probes.number_of_evaluations()


def test_StructuredGrid_Box_vector_statistics_seg(x, V, dirpath):
    origin = [0.25, 0.25, 0.25]                 # origin of box
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # coordinate vectors (scaled in StructuredGrid)
    dL = [0.5, 0.5, 0.5]                        # extent of slice in both directions
    N  = [9, 9, 6]                              # number of points in each direction

    sl = StructuredGrid(V, N, origin, vectors, dL, statistics=True)
    sl(x[0], x[0], x[0])     # probe once
    sl(x[0], x[0], x[0])     # probe once more
    sl.tovtk(0, filename=dirpath+"dump_stats.vtk")
    sl.toh5(0, 1, dirpath+'dump_stats.h5', dtype='float64')
    sl2 = StructuredGrid(V, restart=dirpath+'dump_stats.h5')

    assert sl.dL[0] == sl2.dL[0] and sl.dL[1] == sl2.dL[1] and sl.dL[2] == sl2.dL[2] 
    assert round(sl.arithmetic_mean() - sl2.arithmetic_mean(), 7) == 0
    assert sl.probes.number_of_evaluations() == sl2.probes.number_of_evaluations()
