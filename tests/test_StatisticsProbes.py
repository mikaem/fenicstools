import nose

from dolfin import FunctionSpace, UnitCubeMesh, UnitSquareMesh, interpolate, \
    Expression, VectorFunctionSpace
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
        
    id0, p = probes[0] 
    if len(p) > 0:     
        assert p.number_of_evaluations() == 5
        assert p.value_size() == 5

        mean = p.mean()
        var = p.variance()
        nose.tools.assert_almost_equal(p[0][0], 2.5)
        nose.tools.assert_almost_equal(p[0][4], 0.625)
        nose.tools.assert_almost_equal(p[1][0], 0.5)
        nose.tools.assert_almost_equal(p[1][1], 0.25)
        nose.tools.assert_almost_equal(mean[0], 0.5)
        nose.tools.assert_almost_equal(mean[1], 0.25)
        nose.tools.assert_almost_equal(var[0], 0.25)
        nose.tools.assert_almost_equal(var[1], 0.0625)
        nose.tools.assert_almost_equal(var[2], 0.125)
        
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
        
    id0, p = probes[0] 
    if len(p) > 0:     
        assert p.number_of_evaluations() == 5
        assert p.value_size() == 9

        mean = p.mean()
        var = p.variance()
        nose.tools.assert_almost_equal(p[0][0], 2.5)
        nose.tools.assert_almost_equal(p[0][4], 0.3125)
        nose.tools.assert_almost_equal(p[1][0], 0.5)
        nose.tools.assert_almost_equal(p[1][1], 0.25)
        nose.tools.assert_almost_equal(p[1][2], 0.25)
        nose.tools.assert_almost_equal(mean[0], 0.5)
        nose.tools.assert_almost_equal(mean[1], 0.25)
        nose.tools.assert_almost_equal(mean[2], 0.25)
        nose.tools.assert_almost_equal(var[0], 0.25)
        nose.tools.assert_almost_equal(var[1], 0.0625)
        nose.tools.assert_almost_equal(var[2], 0.0625)
        nose.tools.assert_almost_equal(var[3], 0.125)
        nose.tools.assert_almost_equal(var[4], 0.125)

def test_StatisticsProbes_vector_2D():
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression(('x[0]', 'x[1]')), V)
    x = array([[0.5, 0.25], [0.4, 0.4], [0.3, 0.3]])
    probes = StatisticsProbes(x.flatten(), V)

    for i in range(5):
        probes(u0)
        
    id0, p = probes[0] 
    if len(p) > 0:     
        assert p.number_of_evaluations() == 5
        assert p.value_size() == 5

        mean = p.mean()
        var = p.variance()
        nose.tools.assert_almost_equal(p[0][0], 2.5)
        nose.tools.assert_almost_equal(p[0][4], 0.625)
        nose.tools.assert_almost_equal(p[1][0], 0.5)
        nose.tools.assert_almost_equal(p[1][1], 0.25)
        nose.tools.assert_almost_equal(mean[0], 0.5)
        nose.tools.assert_almost_equal(mean[1], 0.25)
        nose.tools.assert_almost_equal(var[0], 0.25)
        nose.tools.assert_almost_equal(var[1], 0.0625)
        nose.tools.assert_almost_equal(var[2], 0.125)

def test_StatisticsProbes_vector_3D():
    mesh = UnitCubeMesh(4, 4, 4)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    u0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), V)
    x = array([[0.5, 0.25, 0.25], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]])
    probes = StatisticsProbes(x.flatten(), V)

    for i in range(5):
        probes(u0)
        
    id0, p = probes[0] 
    if len(p) > 0:     
        assert p.number_of_evaluations() == 5
        assert p.value_size() == 9

        mean = p.mean()
        var = p.variance()
        nose.tools.assert_almost_equal(p[0][0], 2.5)
        nose.tools.assert_almost_equal(p[0][4], 0.3125)
        nose.tools.assert_almost_equal(p[1][0], 0.5)
        nose.tools.assert_almost_equal(p[1][1], 0.25)
        nose.tools.assert_almost_equal(p[1][2], 0.25)
        nose.tools.assert_almost_equal(mean[0], 0.5)
        nose.tools.assert_almost_equal(mean[1], 0.25)
        nose.tools.assert_almost_equal(mean[2], 0.25)
        nose.tools.assert_almost_equal(var[0], 0.25)
        nose.tools.assert_almost_equal(var[1], 0.0625)
        nose.tools.assert_almost_equal(var[2], 0.0625)
        nose.tools.assert_almost_equal(var[3], 0.125)
        nose.tools.assert_almost_equal(var[4], 0.125)

if __name__ == '__main__':
    nose.run(defaultTest=__name__)