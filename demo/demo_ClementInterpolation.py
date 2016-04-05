from __future__ import division
from fenicstools.ClementInterpolation import clement_interpolate
from dolfin import *


class CIDriver(object):
    '''Driver for ClementInterpolation demos.'''
    def setup(self): raise NotImplementedError

    def pprint(self, args):
        if MPI.rank(mpi_comm_world()) == 0: print args

    def run_case(self, which, with_plot=False):
        '''Show of L2 order of convergence for some predefined test cases.'''
        RED = '\033[1;37;31m%s\033[0m'
        # Setup the test case:
        u0, v0, cases, mesh, get_uv, nrefines = self.setup()
        expr, exact = cases[which]
        # Compute convergence and stuff
        e0, h0, dim0 = None, None, None
        table = ['\t\t'.join(['h', 'e', RED % 'EOC', 'Time', 'len(Ih)', RED % 'Scaling'])]
        self.pprint(table[-1])
        for _ in range(nrefines):
            u, v = get_uv(mesh, u0, v0)

            t = Timer('CI')
            uh, CI = clement_interpolate(expr(u, v), True)
            t = t.stop()
            e = errornorm(exact, uh, 'L2', mesh=mesh)

            h = mesh.hmin()
            dim = uh.function_space().dim()
            if e0 is not None:
                rate = ln(e/e0)/ln(h/h0)
                scale = ln(t/t0)/ln(dim/dim0)
                fmt = ['%3f' % arg for arg in (h, e, t)] + ['%8d' % dim]
                fmt.insert(2, RED % ('%3f' % rate))
                fmt.append(RED % ('%3f' % scale))
                table.append('\t'.join(fmt))
                self.pprint(table[-1])

            e0, h0, t0, dim0 = e, h, t, dim
            mesh = refine(mesh)
    
        self.pprint('Final interpolant has %d dofs' % uh.function_space().dim())
        self.pprint('It took about %gs to construct' % sum(CI.timings()))

        if with_plot and len(uh.ufl_shape) < 2:
            e = interpolate(exact, uh.function_space())
            e.vector().axpy(-1, uh.vector())
            plot(e, title='Error')
            interactive()

        return table


class DemoCI1d(CIDriver):
    '''Demo Clement interpolation in 1d.'''
    def setup(self):
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

        mesh = UnitIntervalMesh(128)

        def get_uv(mesh, u0, v0):
            U = FunctionSpace(mesh, 'CG', 2)
            V = FunctionSpace(mesh, 'CG', 1)
            W = MixedFunctionSpace([U, V])
            w = interpolate(Expression((u0, v0), degree=3), W)
            u, v = w.split()
            return u, v

        nrefines = 9

        return u0, v0, cases, mesh, get_uv, nrefines


class DemoCI2d(CIDriver):
    '''Demo Clement interpolation in 2d.'''
    def setup(self):
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

        mesh = UnitSquareMesh(4, 4)

        def get_uv(mesh, u0, v0):
            U = VectorFunctionSpace(mesh, 'CG', 1)
            u = interpolate(u0, U)
            V = FunctionSpace(mesh, 'CG', 2)
            v = interpolate(v0, V)
            return u, v

        nrefines = 8

        return u0, v0, cases, mesh, get_uv, nrefines


class DemoCI3d(CIDriver):
    '''Demo Clement interpolation in 3d.'''
    def setup(self):
        u0 = Expression(('x[0]*x[0]', 'x[1]*x[1]', 'x[2]*x[2]'), degree=2)
        v0 = None

        cases = {0: (lambda u, v: sin(det(grad(u))),
                 Expression('sin(8*x[0]*x[1]*x[2])', degree=4))}

        mesh = UnitCubeMesh(1, 1, 1)

        def get_uv(mesh, u0, v0):
            U = VectorFunctionSpace(mesh, 'CG', 1)
            u = interpolate(u0, U)
            return u, None

        nrefines = 6

        return u0, v0, cases, mesh, get_uv, nrefines


# ----------------------------------------------------------------------------

if __name__ == '__main__':
    usage = '''
    There are respectively 2, 4 and 1 examples demoing Clement interpolation
    in dims 1 through 3. You can select them individualy by running 
    
        ./demo_ClementInterpolation.py dim, i

    where i is the index of the demo and dim is the spatial dimension. If called
    with 'all' all demos are run. For each example we show off order of convergence 
    of the constructed interpolant(EOC) and the rate at which construction time 
    increases with number of dofs of the interpolant. Note that 1d demos, i.e. 
    i=0, 1, do not work in parallel since they are based on mesh refinement which 
    is only supported in serial.
    '''
    import sys
    assert len(sys.argv) in (2, 3) or len(sys.argv) == 2 and sys.argv[1] == 'all', usage
    
    if sys.argv[1] == 'all':
        demos = [(dim, which) for dim, count in [(0, 2), (1, 4), (2, 1)]
                 for which in range(count)]
    else:
        demos = [map(int, sys.argv[1:])]

    for dim, which in demos: (DemoCI1d, DemoCI2d, DemoCI3d)[dim]().run_case(which)
