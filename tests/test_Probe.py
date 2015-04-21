#!/usr/bin/env py.test

from dolfin import FunctionSpace, UnitCubeMesh, UnitSquareMesh, interpolate, \
                   Expression, mpi_comm_self, VectorFunctionSpace
from fenicstools import *
from numpy import array
import pytest
from fixtures import *


def test_Probe_functionspace_2D(V2_self):
    u0 = interpolate(Expression('x[0]'), V2_self)
    x = array([0.5, 0.5])

    p = Probe(x, V2_self)
    p(u0)
    p(u0)
    assert round(p[0][0] - 0.5, 7) == 0
    assert round(p[1][0] - 0.5, 7) == 0


def test_Probe_functionspace_3D(V3_self):
    u0 = interpolate(Expression('x[0]'), V3_self)
    x = array([0.25, 0.5, 0.5])
    
    p = Probe(x, V3_self)
    p(u0)
    p(u0)
    assert round(p[0][0] - 0.25, 7) == 0
    assert round(p[1][0] - 0.25, 7) == 0


def test_Probe_vectorfunctionspace_2D(VF2_self):
    u0 = interpolate(Expression(('x[0]', 'x[1]')), VF2_self)
    x = array([0.5, 0.75])
    
    p = Probe(x, VF2_self)
    p(u0)
    p(u0)
    assert round(p[0][0] - 0.5, 7) == 0
    assert round(p[1][1] - 0.75, 7) == 0


def test_Probe_vectorfunctionspace_3D(VF3_self):
    u0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), VF3_self)
    x = array([0.25, 0.5, 0.75])
    
    p = Probe(x, VF3_self)
    p(u0)
    p(u0)
    assert round(p[0][0] - 0.25, 7) == 0
    assert round(p[1][1] - 0.50, 7) == 0
    assert round(p[1][2] - 0.75, 7) == 0
