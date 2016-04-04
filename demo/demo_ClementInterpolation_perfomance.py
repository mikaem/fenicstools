from dolfin import *
from fenicstools.ClementInterpolation import clement_interpolate, ClementInterpolant
import time

mesh = UnitSquareMesh(200, 200)
Q = FunctionSpace(mesh, 'DG', 0)
q = Function(Q)
q_vec = q.vector()


def performance_CI(repeats):
    '''Clement interpolation with a precomputed operator.'''
    total = 0
    ci = ClementInterpolant(q)
    for t in range(repeats):
        q_vec[:] = interpolate(Constant(t), Q).vector()

        t0 = time.time()
        v = ci()
        total += time.time() - t0
        
        assert near(q_vec.norm('linf'), v.vector().norm('linf'), 1E-14)

    return repeats, total, total/repeats


def performance_ci(repeats):
    '''On-the-fly Clement interpolation.'''
    total = 0
    for t in range(repeats):
        q_vec[:] = interpolate(Constant(t), Q).vector()

        t0 = time.time()
        v = clement_interpolate(q)
        total += time.time() - t0
        
        assert near(q_vec.norm('linf'), v.vector().norm('linf'), 1E-14)

    return  repeats, total, total/repeats

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    # This program illustrates the speed up of interpolating with the
    # ClementInterpolant vs interpolating with clement_interpolate, i.e. with
    # and withou precomputing stuff. You should observe 10-15x speed up.
    # If you feel like it you can run this script in parallel.

    nrepeats = 20
    ci = performance_ci(repeats=nrepeats)
    CI = performance_CI(repeats=nrepeats)

    if MPI.rank(mpi_comm_world()) == 0:
        print 'clement_interpolate', '%d repeats in %.2f s, %.4f s per repeat' % ci
        print 'ClementInterpolant ', '%d repeats in %.2f s, %.4f s per repeat' % CI
        print 'ClementInterpolant speed up', ci[-1]/CI[-1]
