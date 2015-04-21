#!/usr/bin/env py.test

import pytest
from dolfin import FunctionSpace, UnitCubeMesh, UnitSquareMesh, interpolate, \
                   Expression, VectorFunctionSpace
from fenicstools import *
from numpy import array
from fixtures import *

def test_StatisticsProbe_segregated_2D(V2_self):
    u0 = interpolate(Expression('x[0]'), V2_self)
    v0 = interpolate(Expression('x[1]'), V2_self)
    x = array([0.5, 0.25])

    p = StatisticsProbe(x, V2_self, True)    
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


def test_StatisticsProbe_segregated_3D(V3_self):
    u0 = interpolate(Expression('x[0]'), V3_self)
    v0 = interpolate(Expression('x[1]'), V3_self)
    w0 = interpolate(Expression('x[2]'), V3_self)
    x = array([0.5, 0.25, 0.25])

    p = StatisticsProbe(x, V3_self, True)    
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


def test_StatisticsProbe_vector_2D(VF2_self):
    u0 = interpolate(Expression(('x[0]', 'x[1]')), VF2_self)
    x = array([0.5, 0.25])

    p = StatisticsProbe(x, VF2_self)    
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


def test_StatisticsProbe_vector_3D(VF3_self):
    u0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), VF3_self)
    x = array([0.5, 0.25, 0.25])

    p = StatisticsProbe(x, VF3_self)
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
