from __future__ import division
from dolfin import *
import numpy as np
import ufl


GREEN = '\033[1;37;32m%s\033[0m' 
RED = '\033[1;37;31m%s\033[0m'

class TimerDecorator(object):
    '''Use dolfin.Timer to get execution time.'''
    def __init__(self, name, depth=1):
        self.name = GREEN % name
        self.depth = depth
    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            timer = Timer(self.name)
            ans = f(*args, **kwargs)
            info(' '.join(['\t'*self.depth, self.name, 'done in %.2f s.' % timer.stop()]))
            return ans
        return wrapped_f


class ClementInterpolant(object):
    '''
    This class implements efficient construction of Clement interpolant of an
    UFL-built expression. Here, the Clement interpolant is a CG_1 function over 
    mesh constructed in two steps (See Braess' Finite element book):
        1) For each mesh vertex xj let wj the union of cells that share the vertex 
           (i.e wj is the support of vj - the basis function of CG_1 function
           space such that vj(xj) = 1). Then Qj(expr) is an L2 projection of
           expr into constant field on wj.
        2) Set Ih(expr) = sum_j Qj(expr)vj.
    '''

    @TimerDecorator('precompute Clement interpolant')
    def __init__(self, expr):
        '''For efficient interpolation things are precomuputed here'''
        # Analyze expr and raise if invalid
        terminals = analyze_expr(expr)
        # Analyze shape and raise if expr cannot be represented
        shape = expr.ufl_shape
        analyze_shape(shape)
        # Extract mesh from expr operands and raise if it is not unique or missing
        mesh = extract_mesh(terminals)
        # Compute things for constructing Q
        Q = FunctionSpace(mesh, 'DG', 0)
        q = TestFunction(Q)
        # Need cell volumes for averaging [lhs]
        volumes = assemble(inner(Constant(1), q)*dx)
        # L2 projections [rhs]
        # Scalar, Vectors, Tensors are built from components
        # Translate expression into forms for individual components
        if len(shape) == 0: forms = [inner(expr, q)*dx]
        elif len(shape) == 1: forms = [inner(expr[i], q)*dx for i in range(shape[0])]
        else: forms = [inner(expr[i, j], q)*dx for i in range(shape[0]) for j in range(shape[1])]
        # L2 projections of comps to indiv. cells
        projections = map(assemble, forms)
        # Precompute averaging operator: Interpolant will be built from entries of 
        # projections/volumes in appropriate cells of the patch that supports basis 
        # functions of CG_1. Map of the entries to single dof value is provided by 
        # averaging operator A
        V = FunctionSpace(mesh, 'CG', 1)
        A = construct_averaging_operator(V)
        # Precompute 'mass matrix inverse'
        patch_volumes = Function(V).vector()
        A.mult(volumes, patch_volumes)
        # Awkard poitwise inverse (inverting the mass matrix)
        patch_volumes = as_backend_type(patch_volumes)
        try:
            patch_volumes.vec()[:] = 1./patch_volumes.vec()
        except AttributeError:
            patch_volumes.set_local(1./patch_volumes.get_local())
        patch_volumes.apply('insert')
        # Collect stuff
        self.shape, self.V, self.A, self.patch_volumes, self.projections = \
                shape, V, A, patch_volumes, projections


    @TimerDecorator('compute Clement interpolant')
    def __call__(self):
        '''Return the interpolant.'''
        shape, V, A, patch_volumes, projections = \
                self.shape, self.V, self.A, self.patch_volumes, self.projections
        # The interpolant (scalar, vector, tensor) is build from components
        components = []
        for projection in projections:
            component = Function(V)
            # Compute rhs for L2 patch projection
            A.mult(projection, component.vector()) 
            # Apply the mass matrix inverse
            component.vector()[:] *= patch_volumes
            components.append(component)
        # Finalize the interpolant
        # Scalar has same space as component
        if len(shape) == 0: 
            uh = components.pop()
        # Other ranks
        else:
            W = VectorFunctionSpace(mesh, 'CG', 1, dim=shape[0]) if len(shape) == 1 else\
                TensorFunctionSpace(mesh, 'CG', 1, shape=shape)
            uh = Function(W)
            assign(uh, components)

        return uh

# Workers--

@TimerDecorator('analyze expression', 2)
def analyze_expr(expr):
    '''
    A valid expr for Clement interpolation is defined only in terms of pointwise
    operations on finite element functions.
    '''
    # Elliminate forms
    if isinstance(expr, ufl.Form): raise ValueError('Expression is a form')
    # Elliminate expressions build from Trial/Test functions, FacetNormals 
    terminals = [t for t in ufl.algorithms.traverse_unique_terminals(expr)]
    if any(isinstance(t, (ufl.Argument, ufl.FacetNormal)) for t in terminals):
        raise ValueError('Invalid expression (e.g. has Arguments as operand)')
    # At this point the expression is valid
    return terminals


@TimerDecorator('analyze shape', 2)
def analyze_shape(shape):
    '''
    The shape of expr that UFL can build is arbitrary but we only support
    scalar, rank-1 and rank-2(square) tensors.
    '''
    is_valid = len(shape) < 3 and (shape[0] == shape[1] if len(shape) == 2 else True)
    if not is_valid:
        raise ValueError('Interpolating Expr does not result rank-0, 1, 2 function')


@TimerDecorator('extract mesh', 2)
def extract_mesh(terminals):
    '''Get the common mesh of operands that make the expression.'''
    pairs = []
    for t in terminals:
        try: 
            mesh = t.function_space().mesh()
            pairs.append((mesh.id(), mesh))
        except AttributeError: 
            pass
    ids = set(id_ for id_, _ in pairs)
    # Unique mesh
    if len(ids) == 1: return pairs.pop()[1]
    # Mesh of Nones of multiple
    raise ValueError('Failed to extract mesh: Operands with no or different meshes')


@TimerDecorator('construct averaging operator', 2)
def construct_averaging_operator(V):
    '''
    Avaraging matrix has the following properties: It is a map from DG0 to CG1.
    It has the same sparsity pattern as the mass matrix and in each row the nonzero
    entries are 1. Finally let v \in DG0 then (A*v)_i is the sum of entries of v
    that live on the support of i-th basis function of CG1.
    '''
    mesh = V.mesh()
    Q = FunctionSpace(mesh, 'DG', 0)
    q = TrialFunction(Q)
    v = TestFunction(V)
    tdim = mesh.topology().dim()
    K = CellVolume(mesh)
    dX = dx(metadata={'form_compiler_parameters': {'quadrature_degree': 1,
                                                   'quadrature_scheme': 'vertex'}})
    # This is a nice trick which uses properties of the vertex quadrature to get
    # only ones as nonzero entries.
    # NOTE: Its is designed spec for CG1. In particular does not work CG2 etc so
    # for such spaces a difference construction is required, e.g. rewrite nnz
    # entries of mass matric V, Q to 1. That said CG2 is the highest order where
    # clement interpolation makes sense. With higher ordered the dofs that are
    # interior to cell (or if there are multiple dofs par facet interior) are
    # assigned the same value.
    A = assemble((1./K)*Constant(tdim+1)*inner(v, q)*dX)

    return A

def clement_interpolate(expr):
    '''
    A free function for construting Clement interpolant of an expr. This is
    done by creating instance of ClementInterpolant and applying it. The
    instance is not cached. The function is intended for one time interpolation.
    '''
    ci = ClementInterpolant(expr)
    return ci()


# --- Tests

def test_analyze_extract():
    '''Test quering expressions for Clement interpolation.'''
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    g = Expression('x[0]', degree=1)
    c = Constant(1)
    n = FacetNormal(mesh)
    e = Function(FunctionSpace(mesh, 'CG', 2))
    x = Function(FunctionSpace(UnitIntervalMesh(2), 'CG', 2))
    
    # Raises
    expressions = (u, v, inner(u, v), inner(f, v), dot(grad(f), n),
                   inner(grad(f), grad(v)), inner(f, f)*dx, inner(f, f)*ds)
    count = 0
    for expr in expressions:
        try: analyze_expr(expr)
        except ValueError:
            count += 1
    assert len(expressions) == count

    # Pass analysis
    expressions = (f, grad(f), inner(f, f) + inner(grad(f), grad(f)), inner(f, g)+c, 
                   grad(f)[0], f+g, inner(f, g), c+e, inner(grad(e), grad(f)),
                   x+e, c, g)
    terminals = map(analyze_expr, expressions)

    assert all(extract_mesh(term) for i, term in enumerate(terminals[:9]))

    # Fails to extract
    count = 0
    for term in terminals[9:]:
        try: extract_mesh(term)
        except ValueError:
            count += 1
    assert 3 == count
    print 'TESTS PASSED'


def test_averaging_operator():
    '''Test logic of averaging operator'''
    meshes = (IntervalMesh(3, -1, 30),
              RectangleMesh(Point(-1, -2), Point(2, 4), 4, 8),
              BoxMesh(Point(0, 0, 0), Point(1, 2, 3), 4, 3, 2))
   
    for i, mesh in enumerate(meshes):
        V = FunctionSpace(mesh, 'CG', 1)
        A = construct_averaging_operator(V)
        # Shape
        Q = FunctionSpace(mesh, 'DG', 0)
        assert (A.size(0), A.size(1)) == (V.dim(), Q.dim())
        # All nonzero values are 1
        entries = np.unique(A.array().flatten())
        assert all(near(e, 1, 1E-14) for e in entries[np.abs(entries) > 1E-14])

        # FIXME: Add parallel test for this. I skip it now for it is a bit too
        # involved.
        # The action on a vector of cell volumes should give vector volumes of
        # supports of CG1 functions
        q = TestFunction(Q)
        volumes = assemble(inner(Constant(1), q)*dx)
        # Just check that this is really the volume vector
        va = volumes.array()
        dofmap = Q.dofmap()
        va0 = np.zeros_like(va)
        for cell in cells(mesh): va0[dofmap.cell_dofs(cell.index())[0]] = cell.volume()
        assert np.allclose(va, va0)
        # Compute patch volumes with A
        patch_volumes = Function(V).vector()
        A.mult(volumes, patch_volumes)
        
        # The desired result: patch_volumes
        tdim = i+1
        mesh.init(0, tdim)
        d2v = dof_to_vertex_map(V)
        pv0 = np.array([sum(Cell(mesh, cell).volume()
                            for cell in Vertex(mesh, d2v[dof]).entities(tdim))
                        for dof in range(V.dim())])
        patch_volumes0 = Function(V).vector()
        patch_volumes0.set_local(pv0)
        patch_volumes0.apply('insert')

        patch_volumes -= patch_volumes0
        assert patch_volumes.norm('linf') < 1E-14


    print 'TESTS PASSED'

# --- Demo

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

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    assert len(sys.argv) == 2

    # Test
    if sys.argv[1] == 'test':
        test_analyze_extract()
        test_averaging_operator()
    # Demos
    else:
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
