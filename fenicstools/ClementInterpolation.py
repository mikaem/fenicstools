from itertools import izip
from dolfin import *
import ufl


def clement_interpolate(expr):
    '''
    Construct the Clement interpolant uh of an expr. Here, the Clement interpolant 
    is a CG_1 function over msh constructed in two steps (See Braess' book):
        1) For each mesh vertex xj let wj the union of cells that share the vertex 
           (equiv wj is the support of vj - the basis function of CG_1 function
           space such that vj(xj) = 1). Then Qj(expr) is an L2 projection of
           expr into constant field on wj.
        2) Set Ih(expr) = sum_j Qj(expr)vj
    '''
    # Make sure that the expression is valid
    shape = analyze_expr(expr)
    if shape is None: 
        raise ValueError('Invalid expression: Argument/FacetNormal as operands')
    # The shape of expr that UFL can build is arbitrary but we only support
    # scalar, rank-1 and rank-2(square) tensors
    if not (len(shape) in (0, 1, 2) and 
            (shape[0] == shape[1] if len(shape) == 2 else True)):
        raise ValueError('Interpolating Expr does not result rank-0, 1, 2 function')

    # Try to extract the unique mesh from expr operands
    mesh = extract_mesh(expr)
    if mesh is None:
        raise ValueError('Failed to extract mesh: Operands with no or different meshes')

    # Construct the interpolant
    # Scalar, Vectors, Tensors are all build from CG_1
    V = FunctionSpace(mesh, 'CG', 1)
    # Build patches as map dof -> cells of the the patch
    v2d = vertex_to_dof_map(V)
    patches = []

    tdim = mesh.topology().dim()
    mesh.init(0, tdim)
    for vertex in vertices(mesh):
        patches.append([Cell(mesh, index) for index in vertex.entities(tdim)])

    # FIXME
    cellf = CellFunction('size_t', mesh, 0)
    dx = Measure('dx', domain=mesh, subdomain_id=1, subdomain_data=cellf)

    # Translate expression into forms for individual components of interpolant
    if len(shape) == 0: forms = [expr*dx]
    elif len(shape) == 1: forms = [expr[i]*dx for i in range(shape[0])]
    else: forms = [expr[i, j]*dx for i in range(shape[0]) for j in range(shape[1])]
    
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

# FIXME
def assemble_local(form, cell):
    indicator = form.subdomain_data()[form.ufl_domain()]['cell']
    indicator.set_all(0)
    indicator[cell] = 1
    return assemble(form)


def is_valid(expr):
    '''Operands of the valid expression are only coefficients'''
    return all(not isinstance(t, (ufl.Argument, ufl.FacetNormal))
               for t in ufl.algorithms.traverse_unique_terminals(expr))


def analyze_expr(expr):
    '''
    A valid expr for Clement interpolation is defined only in terms of pointwise
    operations on finite element functions.
    '''
    if hasattr(expr, 'ufl_operands'):
        # Elliminate forms
        if isinstance(expr, ufl.Form): return None
        # Elliminate expressions build from Trial/Test functions, FacetNormals 
        if not is_valid(expr): 
            return None
        # At this point we can get the thing which matters -> the shape
        return expr.ufl_shape
    else:
        return None


def extract_mesh(expr):
    '''Get the common mesh of operands that make the expression.'''
    pairs = []
    for t in ufl.algorithms.traverse_unique_terminals(expr):
        try: 
            mesh = t.function_space().mesh()
            pairs.append((mesh.id(), mesh))
        except AttributeError: 
            pass

    ids = set(id_ for id_, _ in pairs)
    # Unique mesh
    if len(ids) == 1: return pairs.pop()[1]
    # Mesh of Nones of multiple
    return None

# ---

def test_analyze():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    g = Expression('x[0]')
    c = Constant(1)
    n = FacetNormal(mesh)
    
    expressions_false = (u, v, inner(u, v), inner(f, v), dot(grad(f), n),
                         inner(grad(f), grad(v)), inner(f, f)*dx, inner(f, f)*ds)
    expressions_true = (f, g, c, grad(f), inner(f, f) + inner(grad(f), grad(f)),
                        inner(f, g)+c, grad(f)[0])

    assert all(analyze_expr(e) is None for e in expressions_false)
    assert all(analyze_expr(e) is not None for e in expressions_true)


def test_extract():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)
    g = Function(V)
    c = Constant(1)
    e = Function(FunctionSpace(mesh, 'CG', 2))
    x = Function(FunctionSpace(UnitIntervalMesh(2), 'CG', 2))

    expressions_true = (f+g, inner(f, g), c+e, inner(grad(e), grad(f)), f)
    expressions_false = (c, x+e)

    assert all(extract_mesh(e) is None for e in expressions_false)
    assert all(extract_mesh(e) is not None for e in expressions_true)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # test_analyze()
    # test_extract()

    mesh = UnitSquareMesh(20, 20)
    V = FunctionSpace(mesh, 'CG', 1)
    u = interpolate(Expression('x[0]*x[0]'), V)
    expr = inner(div(grad(u)), u)*dx
    uh = clement_interpolate(expr)

    plot(uh)
    interactive()
