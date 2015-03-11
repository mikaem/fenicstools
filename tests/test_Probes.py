#!/usr/bin/env py.test

import pytest
from dolfin import FunctionSpace, UnitCubeMesh, UnitSquareMesh, interpolate, \
                   Expression, MPI, mpi_comm_world, VectorFunctionSpace
from fenicstools import *
from numpy import array, load
from fixtures import *


def test_Probes_functionspace_2D(V2):
    u0 = interpolate(Expression('x[0]'), V2)
    x = array([[0.5, 0.5], [0.4, 0.4], [0.3, 0.3]])

    p = Probes(x.flatten(), V2)
    # Probe twice
    p(u0)
    p(u0)
    
    # Check both snapshots
    p0 = p.array(N=0)
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0] - 0.5, 7) == 0
        assert round(p0[1] - 0.4, 7) == 0
        assert round(p0[2] - 0.3, 7) == 0
    p0 = p.array(N=1)
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0] - 0.5, 7) == 0
        assert round(p0[1] - 0.4, 7) == 0
        assert round(p0[2] - 0.3, 7) == 0


def test_Probes_functionspace_3D(V3, dirpath):
    u0 = interpolate(Expression('x[0]'), V3)
    x = array([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]])
    
    p = Probes(x.flatten(), V3)
    
    # Probe twice
    p(u0)
    p(u0)
    
    # Check both snapshots
    p0 = p.array(N=0)
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0] - 0.5, 7) == 0
        assert round(p0[1] - 0.4, 7) == 0
        assert round(p0[2] - 0.3, 7) == 0
    p0 = p.array(N=1)
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0] - 0.5, 7) == 0
        assert round(p0[1] - 0.4, 7) == 0
        assert round(p0[2] - 0.3, 7) == 0
        
    p0 = p.array(filename=dirpath+'dump')
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0, 0] - 0.5, 7) == 0
        assert round(p0[1, 1] - 0.4, 7) == 0
        assert round(p0[2, 1] - 0.3, 7) == 0
        
        f = open(dirpath+'dump_all.probes', 'r')
        p1 = load(f)
        assert round(p1[0, 0, 0] - 0.5, 7) == 0
        assert round(p1[1, 0, 1] - 0.4, 7) == 0
        assert round(p1[2, 0, 1] - 0.3, 7) == 0


def test_Probes_vectorfunctionspace_2D(VF2, dirpath):
    u0 = interpolate(Expression(('x[0]', 'x[1]')), VF2)
    x = array([[0.5, 0.5], [0.4, 0.4], [0.3, 0.3]])
    
    p = Probes(x.flatten(), VF2)

    # Probe twice
    p(u0)
    p(u0)
    
    # Check both snapshots
    p0 = p.array(N=0)
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0, 0] - 0.5, 7) == 0
        assert round(p0[1, 1] - 0.4, 7) == 0
        assert round(p0[2, 1] - 0.3, 7) == 0
    p0 = p.array(N=1)
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0, 0] - 0.5, 7) == 0
        assert round(p0[1, 0] - 0.4, 7) == 0
        assert round(p0[2, 1] - 0.3, 7) == 0

    p0 = p.array(filename=dirpath+'dumpvector2D')
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0, 0, 0] - 0.5, 7) == 0
        assert round(p0[1, 1, 1] - 0.4, 7) == 0
        assert round(p0[2, 0, 1] - 0.3, 7) == 0
        
        f = open(dirpath+'dumpvector2D_all.probes', 'r')
        p1 = load(f)
        assert round(p1[0, 0, 0] - 0.5, 7) == 0
        assert round(p1[1, 1, 0] - 0.4, 7) == 0
        assert round(p1[2, 1, 1] - 0.3, 7) == 0


def test_Probes_vectorfunctionspace_3D(VF3, dirpath):
    u0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), VF3)
    x = array([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]])
    
    p = Probes(x.flatten(), VF3)
    # Probe twice
    p(u0)
    p(u0)
    
    # Check both snapshots
    p0 = p.array(N=0)
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0, 0] - 0.5, 7) == 0
        assert round(p0[1, 1] - 0.4, 7) == 0
        assert round(p0[2, 2] - 0.3, 7) == 0
    p0 = p.array(N=1)
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0, 0] - 0.5, 7) == 0
        assert round(p0[1, 1] - 0.4, 7) == 0
        assert round(p0[2, 2] - 0.3, 7) == 0
        
    p0 = p.array(filename=dirpath+'dumpvector3D')
    if MPI.rank(mpi_comm_world()) == 0:
        assert round(p0[0, 0, 0] - 0.5, 7) == 0
        assert round(p0[1, 1, 0] - 0.4, 7) == 0
        assert round(p0[2, 1, 0] - 0.3, 7) == 0
        
        f = open(dirpath+'dumpvector3D_all.probes', 'r')
        p1 = load(f)
        assert round(p1[0, 0, 0] - 0.5, 7) == 0
        assert round(p1[1, 1, 0] - 0.4, 7) == 0
        assert round(p1[2, 1, 1] - 0.3, 7) == 0
