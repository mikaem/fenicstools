#!/usr/bin/env py.test

import pytest
from dolfin import FunctionSpace, UnitSquareMesh, UnitCubeMesh, \
                   interpolate, Function, Expression
from fenicstools import weighted_gradient_matrix
from fixtures import *

def test_WeightedGradient(V2):
    expr = "+".join(["%d*x[%d]" % (i+1,i) for i in range(2)])
    u = interpolate(Expression(expr), V2)
    du = Function(V2)
    wx = weighted_gradient_matrix(V2.mesh(), tuple(range(2)))
    for i in range(2):
        du.vector()[:] = wx[i] * u.vector()
        assert round(du.vector().min() - (i+1), 7) == 0
