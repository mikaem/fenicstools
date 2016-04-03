#!/usr/bin/env py.test
from dolfin import *
import fenicstools.ClementInterpolation as ci
import numpy as np
import pytest


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
        try: ci._analyze_expr(expr)
        except ValueError:
            count += 1
    assert len(expressions) == count

    # Pass analysis
    expressions = (f, grad(f), inner(f, f) + inner(grad(f), grad(f)), inner(f, g)+c, 
                   grad(f)[0], f+g, inner(f, g), c+e, inner(grad(e), grad(f)),
                   x+e, c, g)
    terminals = map(ci._analyze_expr, expressions)

    assert all(ci._extract_mesh(term) for i, term in enumerate(terminals[:9]))

    # Fails to extract
    count = 0
    for term in terminals[9:]:
        try: ci._extract_mesh(term)
        except ValueError:
            count += 1
    assert 3 == count


def test_averaging_operator():
    '''Test logic of averaging operator'''
    meshes = (IntervalMesh(3, -1, 30),
              RectangleMesh(Point(-1, -2), Point(2, 4), 4, 8),
              BoxMesh(Point(0, 0, 0), Point(1, 2, 3), 4, 3, 2))
   
    for i, mesh in enumerate(meshes):
        V = FunctionSpace(mesh, 'CG', 1)
        A = ci._construct_averaging_operator(V)
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
