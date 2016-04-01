from itertools import izip
from dolfin import *
import ufl


def clement_interpolate(expr):
    '''Construct the Clement interpolant of expr'''
    # Make sure that the expression is valid
    try:
        terminals = analyze_expr(expr)
    except ValueError as e:
        print e
        return None

    # Now the shape can be extracted
    shape = expr.ufl_shape
    try:
        analyze_shape(shape)
    except ValueError as e:
        print e
        return None

    # Extract the unique mesh from expr operands
    try:
        mesh = extract_mesh(terminals)
    except ValueError as e:
        print e
        return None

    # Construct the interpolant
    return clement_interpolant(expr, shape, mesh)

# --- Implementation

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


def analyze_shape(shape):
    '''
    The shape of expr that UFL can build is arbitrary but we only support
    scalar, rank-1 and rank-2(square) tensors.
    '''
    is_valid = len(shape) < 3 and (shape[0] == shape[1] if len(shape) == 2 else True)
    if not is_valid:
        raise ValueError('Interpolating Expr does not result rank-0, 1, 2 function')


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


def clement_interpolant(expr, shape, mesh):
    '''
    Here, the Clement interpolant is a CG_1 function over msh constructed in 
    two steps (See Braess' book):
        1) For each mesh vertex xj let wj the union of cells that share the vertex 
           (i.e wj is the support of vj - the basis function of CG_1 function
           space such that vj(xj) = 1). Then Qj(expr) is an L2 projection of
           expr into constant field on wj.
        2) Set Ih(expr) = sum_j Qj(expr)vj
    '''
    # Scalar, Vectors, Tensors are all build from CG_1
    V = FunctionSpace(mesh, 'CG', 1)
    # Build patches as map dof -> cells of the the patch
    # FIXME parallel?
    v2d = vertex_to_dof_map(V)
    patches = []

    tdim = mesh.topology().dim()
    mesh.init(0, tdim)
    for vertex in vertices(mesh):
        patches.append([Cell(mesh, index) for index in vertex.entities(tdim)])

    # FIXME modif when dolfin.assemble_local is available
    cellf = CellFunction('size_t', mesh, 0)
    dX = Measure('dx', domain=mesh, subdomain_id=1, subdomain_data=cellf)

    # Translate expression into forms for individual components of interpolant
    if len(shape) == 0: forms = [expr*dX]
    elif len(shape) == 1: forms = [expr[i]*dX for i in range(shape[0])]
    else: forms = [expr[i, j]*dX for i in range(shape[0]) for j in range(shape[1])]
    
    components = []
    # Build components of interpolant. Combines 1, 2
    for form in forms:
        comp = Function(V)
        vec = comp.vector().array()
        for dof, patch in izip(v2d, patches):
            m = sum(cell.volume() for cell in patch)
            b = sum(assemble_local(form, cell) for cell in patch)
            vec[dof] = b/m
        comp.vector().set_local(vec)
        comp.vector().apply('insert')
        components.append(comp)

    # Finalize the interpolant
    # Scalar has same space as component
    if len(shape) == 0: 
        uh = components.pop()
    # Other ranks
    else:
        W = VectorFunctionSpace(mesh, 'CG', 1) if len(shape) == 1 else\
            TensorFunctionSpace(mesh, 'CG', 1)
        uh = Function(W)
        assign(uh, components)

    return uh

# FIXME remove when dolfin.assemble_local is available
def assemble_local(form, cell):
    indicator = form.subdomain_data()[form.ufl_domain()]['cell']
    indicator.set_all(0)
    indicator[cell] = 1
    return assemble(form)

# --- Tests

def test_analyze_extract():
    '''Test quering expressions for Clement interpolation.'''
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    g = Expression('x[0]')
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

# --- Demo

def demo_ci(which, mesh='uniform', with_plot=False):
    '''Show of L2 order of convergence for some predefined test cases.'''
    # NOTE: The interpolant is CG1 and first order in L2 is expected if the
    # expression is bounded in H1 norm. In the examples order 1.5 is observed on
    # both meshes
    from math import log as ln
    import mshr

    u0 = Expression(('x[0]', 'x[1]'))
    v0 = Expression('x[0]*x[1]') 

    cases = {0: (lambda u, v: sin(inner(u, grad(v))), Expression('sin(2*x[0]*x[1])')),
             1: (lambda u, v: outer(u, grad(v)), Expression((('x[0]*x[1]', 'x[0]*x[0]'),
                                                             ('x[1]*x[1]', 'x[0]*x[1]')))),
             2: (lambda u, v: div(outer(u, grad(v))), Expression(('x[1]', 'x[0]'))),
             3: (lambda u, v: tr(outer(u, grad(v))), Expression('2*x[0]*x[1]'))}

    expr, exact = cases[which]
    if mesh == 'uniform': mesh = UnitSquareMesh(4, 4)
    else: mesh = mshr.generate_mesh(mshr.Rectangle(Point(0, 0), Point(1, 1)), 3)

    e0, h0 = None, None
    print 'h\t\te\t\tEOC'
    for _ in range(5):
        U = VectorFunctionSpace(mesh, 'CG', 1)
        u = interpolate(u0, U)

        V = FunctionSpace(mesh, 'CG', 2)
        v = interpolate(v0, V)

        uh = clement_interpolate(expr(u, v))
        e = errornorm(exact, uh, 'L2', mesh=mesh)

        h = mesh.hmin()
        if e0 is not None:
            print '\t'.join(('%3f' % arg for arg in (h, e, ln(e/e0)/ln(h/h0))))

        e0, h0 = e, h
        mesh = refine(mesh)

    if with_plot and len(uh.ufl_shape) < 2:
        e = interpolate(exact, uh.function_space())
        e.vector().axpy(-1, uh.vector())
        plot(e, title='Error')
        interactive()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    usage = 'Select -1 for test or 0, 1, 2, 3 ["uniform"|"foo"] [1|0] for demos'
    assert len(sys.argv) > 1, usage 

    which = int(sys.argv[1])
    if which == -1:
        test_analyze_extract()
    else:
        mesh = sys.argv[2]
        with_plot = len(sys.argv) == 4 and bool(sys.argv[3])
        demo_ci(which, mesh=mesh, with_plot=with_plot)
