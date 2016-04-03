from __future__ import division
from fenicstools import clement_interpolate
import numpy as np
from dolfin import *


RED = '\033[1;37;31m%s\033[0m'

def demo_ci_1d(which, mesh='uniform', with_plot=False):
    '''Show of L2 order of convergence for some predefined test cases in 1d.'''
    # NOTE: The interpolant is CG1 and first order in L2 is expected if the
    # expression is bounded in H1 norm. In the examples order 1.5 is observed on
    # both meshes
    from math import log as ln

    u0 = 'sin(4*pi*x[0])'
    v0 = 'x[0]*x[0]'

    cases = {0: (lambda u, v: inner(Dx(u, 0), Dx(u, 0)) + v,
                 Expression('16*pi*pi*cos(4*pi*x[0])*cos(4*pi*x[0])+x[0]*x[0]', degree=8)),
             1: (lambda u, v: inner(as_vector((u, Dx(u, 0), Dx(Dx(u, 0), 0))),
                                    as_vector((u, Dx(u, 0), Dx(Dx(u, 0), 0))))+\
                              inner(Constant(0), v),
                 Expression('''sin(4*pi*x[0])*sin(4*pi*x[0])+
                               16*pi*pi*cos(4*pi*x[0])*cos(4*pi*x[0])+
                               256*pi*pi*pi*pi*sin(4*pi*x[0])*sin(4*pi*x[0])''', degree=8))
             }

    expr, exact = cases[which]
    mesh = UnitIntervalMesh(128)
    if not mesh == 'uniform':
        mesh.coordinates()[:] = np.sin(mesh.coordinates()[:])

    e0, h0, dim0 = None, None, None
    table = ['\t\t'.join(['h', 'e', RED % 'EOC', 'Time', 'len(Ih)', RED % 'Scaling'])]
    print table[-1] 
    for _ in range(8):
        U = FunctionSpace(mesh, 'CG', 2)
        V = FunctionSpace(mesh, 'CG', 1)
        W = MixedFunctionSpace([U, V])
        w = interpolate(Expression((u0, v0), degree=3), W)
        u, v = w.split()
        
        # How long it takes to construct the interpolant
        timer = Timer('CI')
        uh = clement_interpolate(expr(u, v))
        t = timer.stop()
        # Error
        e = errornorm(exact, uh, 'L2', mesh=mesh)

        h = mesh.hmin()
        dim = uh.function_space().dim()
        if e0 is not None:
            rate = ln(e/e0)/ln(h/h0)
            scale = ln(t/t0)/ln(dim/dim0)
            fmt = ['%3f' % arg for arg in (h, e, t, dim)]
            fmt.insert(2, RED % ('%3f' % rate))
            fmt.append(RED % ('%3f' % scale))
            table.append('\t'.join(fmt))
            print table[-1]
            

        e0, h0, t0, dim0 = e, h, t, dim
        mesh = refine(mesh)
    
    print 'Final interpolant has %d dofs' % uh.function_space().dim()

    if with_plot and len(uh.ufl_shape) < 2:
        e = interpolate(exact, uh.function_space())
        e.vector().axpy(-1, uh.vector())
        plot(e, title='Error')
        interactive()
    
    return table


def demo_ci_2d(which, mesh='uniform', with_plot=False):
    '''Show of L2 order of convergence for some predefined test cases in 2d.'''
    # NOTE: The interpolant is CG1 and first order in L2 is expected if the
    # expression is bounded in H1 norm. In the examples order 1.5 is observed on
    # both meshes
    from math import log as ln
    import mshr

    u0 = Expression(('x[0]', 'x[1]'), degree=1)
    v0 = Expression('x[0]*x[1]', degree=2) 

    cases = {0: (lambda u, v: sin(inner(u, grad(v))),
                 Expression('sin(2*x[0]*x[1])', degree=4)),
             1: (lambda u, v: outer(u, grad(v)),
                 Expression((('x[0]*x[1]', 'x[0]*x[0]'),
                            ('x[1]*x[1]', 'x[0]*x[1]')), degree=4)),
             2: (lambda u, v: div(outer(u, grad(v))),
                 Expression(('x[1]', 'x[0]'), degree=4)),
             3: (lambda u, v: tr(outer(u, grad(v))),
                 Expression('2*x[0]*x[1]', degree=4))}

    expr, exact = cases[which]
    if mesh == 'uniform': mesh = UnitSquareMesh(4, 4)
    else: mesh = mshr.generate_mesh(mshr.Rectangle(Point(0, 0), Point(1, 1)), 3)

    e0, h0, dim0 = None, None, None
    table = ['\t\t'.join(['h', 'e', RED % 'EOC', 'Time', 'len(Ih)', RED % 'Scaling'])]
    print table[-1]
    for _ in range(8):
        U = VectorFunctionSpace(mesh, 'CG', 1)
        u = interpolate(u0, U)

        V = FunctionSpace(mesh, 'CG', 2)
        v = interpolate(v0, V)
        
        # How long it takes to construct the interpolant
        timer = Timer('CI')
        uh = clement_interpolate(expr(u, v))
        t = timer.stop()
        # Error
        e = errornorm(exact, uh, 'L2', mesh=mesh)

        h = mesh.hmin()
        dim = uh.function_space().dim()
        if e0 is not None:
            rate = ln(e/e0)/ln(h/h0)
            scale = ln(t/t0)/ln(dim/dim0)
            fmt = ['%3f' % arg for arg in (h, e, t, dim)]
            fmt.insert(2, RED % ('%3f' % rate))
            fmt.append(RED % ('%3f' % scale))
            table.append('\t'.join(fmt))
            print table[-1]

        e0, h0, t0, dim0 = e, h, t, dim
        mesh = refine(mesh)
    
    print 'Final interpolant has %d dofs' % uh.function_space().dim()

    if with_plot and len(uh.ufl_shape) < 2:
        e = interpolate(exact, uh.function_space())
        e.vector().axpy(-1, uh.vector())
        plot(e, title='Error')
        interactive()
    
    return table


def demo_ci_3d(which, mesh='uniform', with_plot=False):
    '''Show of L2 order of convergence for some predefined test cases in 3d.'''
    # NOTE: The interpolant is CG1 and first order in L2 is expected if the
    # expression is bounded in H1 norm. In the examples order 1.5 is observed on
    # both meshes
    from math import log as ln
    import mshr

    u0 = Expression(('x[0]*x[0]', 'x[1]*x[1]', 'x[2]*x[2]'), degree=2)

    cases = {0: (lambda u: sin(det(grad(u))),
                 Expression('sin(8*x[0]*x[1]*x[2])', degree=4))}

    expr, exact = cases[which]
    if not mesh == 'uniform': 
        pass
    mesh = UnitCubeMesh(1, 1, 1)
    # NOTE: I ignore mshr mesh for it seems that the mesh can be degenerate

    e0, h0, dim0 = None, None, None
    table = ['\t\t'.join(['h', 'e', RED % 'EOC', 'Time', 'len(Ih)', RED % 'Scaling'])]
    print table[-1]
    for _ in range(6):
        U = VectorFunctionSpace(mesh, 'CG', 1)
        u = interpolate(u0, U)

        # How long it takes to construct the interpolant
        timer = Timer('CI')
        uh = clement_interpolate(expr(u))
        t = timer.stop()
        # Error
        e = errornorm(exact, uh, 'L2', mesh=mesh)

        h = mesh.hmin()
        dim = uh.function_space().dim()
        if e0 is not None:
            rate = ln(e/e0)/ln(h/h0)
            scale = ln(t/t0)/ln(dim/dim0)
            fmt = ['%3f' % arg for arg in (h, e, t, dim)]
            fmt.insert(2, RED % ('%3f' % rate))
            fmt.append(RED % ('%3f' % scale))
            table.append('\t'.join(fmt))
            print table[-1]

        e0, h0, t0, dim0 = e, h, t, dim
        mesh = refine(mesh)
    
    print 'Final interpolant has %d dofs' % uh.function_space().dim()

    if with_plot and len(uh.ufl_shape) < 2:
        e = interpolate(exact, uh.function_space())
        e.vector().axpy(-1, uh.vector())
        plot(e, title='Error')
        interactive()

    return table

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    usage = '''
    There are respectively 2, 4 and 1 examples demoing Clement interpolation
    in dims 1 through 3. You can select them individualy by running 
    
        ./demo_ClementInterpolation.py i

    where i is the index of the demo. If i == 'all' all demos are run. For each
    example we show off order of convergence of the constructed interpolant(EOC)
    and the rate at which construction time increases with number of dofs of the
    interpolant.
    '''
    import sys
    assert len(sys.argv) == 2, usage

    from functools import partial
    mesh = 'uniform'
    with_plot = False
    spec = lambda fi: partial(fi[0], which=fi[1], mesh=mesh, with_plot=with_plot)
    
    demos = map(spec, 
                [(demo_ci_1d, i) for i in range(2)] + \
                [(demo_ci_2d, i) for i in range(4)] + \
                [(demo_ci_3d, i) for i in range(1)])

    if not sys.argv[1] == 'all': demos = [demos[int(sys.argv[1])]]

    for i, demo in enumerate(demos):
        table = demo()
        print '-'*40, 'Demo', i , '-'*40
        for row in table: print row
        print '-'*79

    print list_timings(TimingClear_keep, [TimingType_wall, TimingType_system]) 
