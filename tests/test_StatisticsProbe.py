#!/usr/bin/env py.test

import pytest
from dolfin import FunctionSpace, UnitCubeMesh, UnitSquareMesh, interpolate, \
                   Expression, mpi_comm_self, VectorFunctionSpace
from fenicstools import *
from numpy import array
from fixtures import *

def test_StatisticsProbe_segregated_2D(V2):
    u0 = interpolate(Expression('x[0]'), V2)
    v0 = interpolate(Expression('x[1]'), V2)
    x = array([0.5, 0.25])

    p = StatisticsProbe(x, V2, True)    
    for i in range(5):
        p(u0, v0)
        
    assert p.number_of_evaluations() == 5
    assert p.value_size() == 5

    mean = p.mean()
    var = p.variance()
    assert round(p[0][0] - 2.5, 7) == 0
    assert round(p[0][4] - 0.625, 7) == 0
    assert round(p[1][0] - 0.5, 7) == 0
    assert round(p[1][1] - 0.25, 7) == 0
    assert round(mean[0] - 0.5, 7) == 0
    assert round(mean[1] - 0.25, 7) == 0
    assert round(var[0] - 0.25, 7) == 0
    assert round(var[1] - 0.0625, 7) == 0
    assert round(var[2] - 0.125, 7) == 0


def test_StatisticsProbe_segregated_3D(V3):
    u0 = interpolate(Expression('x[0]'), V3)
    v0 = interpolate(Expression('x[1]'), V3)
    w0 = interpolate(Expression('x[2]'), V3)
    x = array([0.5, 0.25, 0.25])

    p = StatisticsProbe(x, V3, True)    
    for i in range(5):
        p(u0, v0, w0)
        
    assert p.number_of_evaluations() == 5
    assert p.value_size() == 9

    mean = p.mean()
    var = p.variance()
    assert round(p[0][0] - 2.5, 7) == 0
    assert round(p[0][4] - 0.3125, 7) == 0
    assert round(p[1][0] - 0.5, 7) == 0
    assert round(p[1][1] - 0.25, 7) == 0
    assert round(p[1][2] - 0.25, 7) == 0
    assert round(mean[0] - 0.5, 7) == 0
    assert round(mean[1] - 0.25, 7) == 0
    assert round(mean[2] - 0.25, 7) == 0
    assert round(var[0] - 0.25, 7) == 0
    assert round(var[1] - 0.0625, 7) == 0
    assert round(var[2] - 0.0625, 7) == 0
    assert round(var[3] - 0.125, 7) == 0
    assert round(var[4] - 0.125, 7) == 0


def test_StatisticsProbe_vector_2D(VF2):
    u0 = interpolate(Expression(('x[0]', 'x[1]')), VF2)
    x = array([0.5, 0.25])

    p = StatisticsProbe(x, VF2)    
    for i in range(5):
        p(u0)
        
    assert p.number_of_evaluations() == 5
    assert p.value_size() == 5

    mean = p.mean()
    var = p.variance()

    assert round(p[0][0] - 2.5, 7) == 0
    assert round(p[0][4] - 0.625, 7) == 0
    assert round(p[1][0] - 0.5, 7) == 0
    assert round(p[1][1] - 0.25, 7) == 0
    assert round(mean[0] - 0.5, 7) == 0
    assert round(mean[1] - 0.25, 7) == 0
    assert round(var[0] - 0.25, 7) == 0
    assert round(var[1] - 0.0625, 7) == 0
    assert round(var[2] - 0.125, 7) == 0


def test_StatisticsProbe_vector_3D(VF3):
    u0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), VF3)
    x = array([0.5, 0.25, 0.25])

    p = StatisticsProbe(x, VF3)
    for i in range(5):
        p(u0)

    assert p.number_of_evaluations() == 5
    assert p.value_size() == 9

    mean = p.mean()
    var = p.variance()

    assert round(p[0][0] - 2.5, 7) == 0
    assert round(p[0][4] - 0.3125, 7) == 0
    assert round(p[1][0] - 0.5, 7) == 0
    assert round(p[1][1] - 0.25, 7) == 0
    assert round(p[1][2] - 0.25, 7) == 0
    assert round(mean[0] - 0.5, 7) == 0
    assert round(mean[1] - 0.25, 7) == 0
    assert round(mean[2] - 0.25, 7) == 0
    assert round(var[0] - 0.25, 7) == 0
    assert round(var[1] - 0.0625, 7) == 0
    assert round(var[2] - 0.0625, 7) == 0
    assert round(var[3] - 0.125, 7) == 0
    assert round(var[4] - 0.125, 7) == 0
