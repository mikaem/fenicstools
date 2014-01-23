__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
This module contains functionality for efficiently probing a Function many times. 
"""
from dolfin import *
from numpy import alltrue, cos, zeros, array, repeat, squeeze, argmax, cumsum, sum, count_nonzero, reshape, resize, linspace, abs, sign, all, float32
from numpy.linalg import norm as numpy_norm
try:
    from scitools.std import surfc
    from scitools.basics import meshgrid
except:
    pass
import pyvtk, os, copy, cPickle, h5py, inspect
from mpi4py import MPI as nMPI
comm = nMPI.COMM_WORLD

# Compile Probe C++ code
def strip_essential_code(filenames):
    code = ""
    for name in filenames:
        f = open(name, 'r').read()
        code += f[f.find("namespace dolfin\n{\n"):f.find("#endif")]
    return code

dolfin_folder = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), "../Probe"))
sources = ["Probe.cpp", "Probes.cpp", "StatisticsProbe.cpp", "StatisticsProbes.cpp"]
headers = map(lambda x: os.path.join(dolfin_folder, x), ['Probe.h', 'Probes.h', 'StatisticsProbe.h', 'StatisticsProbes.h'])
code = strip_essential_code(headers)
compiled_module = compile_extension_module(code=code, source_directory=os.path.abspath(dolfin_folder),
                                           sources=sources, include_dirs=[".", os.path.abspath(dolfin_folder)])

# Give the compiled classes some additional pythonic functionality
class Probe(compiled_module.Probe):
    
    def __call__(self, *args):
        return self.eval(*args)

    def __len__(self):
        return self.value_size()
    
    def __getitem__(self, i):
        return self.get_probe_at_snapshot(i)

class Probes(compiled_module.Probes):

    def __call__(self, *args):
        return self.eval(*args)
        
    def __len__(self):
        return self.local_size()

    def __iter__(self): 
        self.i = 0
        return self

    def __getitem__(self, i):
        return self.get_probe_id(i), self.get_probe(i)

    def next(self):
        try:
            p =  self[self.i]
        except:
            raise StopIteration
        self.i += 1
        return p    

    def array(self, N=None, filename=None, component=None, root=0):
        """Dump data to numpy format on root processor for all or one snapshot."""
        is_root = comm.Get_rank() == root
        size = self.get_total_number_probes() if is_root else len(self)
        comp = self.value_size() if component is None else 1
        if not N is None:
            z  = zeros((size, comp))
        else:
            z  = zeros((size, comp, self.number_of_evaluations()))
        
        # Get all values
        if len(self) > 0: 
            if not N is None:
                for k in range(comp):
                    if is_root:
                        ids = self.get_probe_ids()
                        z[ids, k] = self.get_probes_component_and_snapshot(k, N)
                    else:
                        z[:, k] = self.get_probes_component_and_snapshot(k, N)
            else:                
                for i, (index, probe) in enumerate(self):
                    j = index if is_root else i
                    if not N is None:
                        z[j, :] = probe.get_probe_at_snapshot(N)
                    else:
                        for k in range(self.value_size()):
                            z[j, k, :] = probe.get_probe_sub(k)
                        
        # Collect values on root
        recvfrom = comm.gather(len(self), root=root)
        if is_root:
            for j, k in enumerate(recvfrom):                
                if comm.Get_rank() != j:
                    ids = comm.recv(source=j, tag=101)
                    z0 = comm.recv(source=j, tag=102)
                    z[ids, :] = z0[:, :]
        else:
            ids = self.get_probe_ids()
            comm.send(ids, dest=root, tag=101)
            comm.send(z, dest=root, tag=102)
            
        if is_root:
            if filename:
                if not N is None:
                    z.dump(filename+"_snapshot_"+str(N)+".probes")
                else:
                    z.dump(filename+"_all.probes")
            return squeeze(z)

class StatisticsProbe(compiled_module.StatisticsProbe):
    
    def __call__(self, *args):
        return self.eval(*args)

    def __len__(self):
        return self.value_size()
    
    def __getitem__(self, i):
        assert(i < 2)
        return self.get_probe_at_snapshot(i)

class StatisticsProbes(compiled_module.StatisticsProbes):

    def __call__(self, *args):
        return self.eval(*args)
        
    def __len__(self):
        return self.local_size()

    def __iter__(self): 
        self.i = 0
        return self

    def __getitem__(self, i):
        return self.get_probe_id(i), self.get_probe(i)

    def next(self):
        try:
            p = self[self.i]
        except:
            raise StopIteration
        self.i += 1
        return p   
        
    def array(self, N=0, filename=None, component=None, root=0):
        """Dump data to numpy format on root processor."""
        assert(N == 0 or N == 1)
        is_root = comm.Get_rank() == root
        size = self.get_total_number_probes() if is_root else len(self)
        comp = self.value_size() if component is None else 1        
        z  = zeros((size, comp))
        
        # Retrieve all values
        if len(self) > 0: 
            for k in range(comp):
                if is_root:
                    ids = self.get_probe_ids()
                    z[ids, k] = self.get_probes_component_and_snapshot(k, N)
                else:
                    z[:, k] = self.get_probes_component_and_snapshot(k, N)
                     
        # Collect on root
        recvfrom = comm.gather(len(self), root=root)
        if is_root:
            for j, k in enumerate(recvfrom):                
                if comm.Get_rank() != j:
                    ids = comm.recv(source=j, tag=101)
                    z0 = comm.recv(source=j, tag=102)
                    z[ids, :] = z0[:, :]
        else:
            ids = self.get_probe_ids()
            comm.send(ids, dest=root, tag=101)
            comm.send(z, dest=root, tag=102)
            
        if is_root:
            if filename:
                z.dump(filename+"_statistics.probes")
            return squeeze(z)

class StructuredGrid:
    """A Structured grid of probe points. 
    
    A slice of a 3D (possibly 2D if needed) domain can be created with any 
    orientation using two tangent vectors to span a local coordinatesystem. 
    Likewise, a Box in 3D is created by supplying three basis vectors.
    
          dims = [N1, N2 (, N3)] number of points in each direction
          origin = (x, y, z) origin of slice 
          dX = [[dx1, dy1, dz1], [dx2, dy2, dz2] (, [dx3, dy3, dz3])] tangent vectors (need not be orthogonal)
          dL = [dL1, dL2 (, dL3)] extent of each direction
          V  = FunctionSpace to probe in
          
          statistics = True  => Compute statistics in probe points (mean/covariance).
                       False => Store instantaneous snapshots of probepoints. 
          
          restart = hdf5-file => Restart probe from previous computations.
          
    """
    def __init__(self, V, dims=None, origin=None, dX=None, dL=None, statistics=False, restart=False, tstep=None):
        if restart:
            statistics = self.read_grid(restart)
        else:
            self.dims = dims
            self.dL, self.dX = array(dL, float), array(dX, float)
            self.origin = origin
            self.dx, self.dy, self.dz = self.create_coordinate_vectors()
            
        self.initialize_probes(statistics, V)
        if restart:
            self.set_data_from_file(filename=restart, tstep=tstep)
                 
    def initialize_probes(self, statistics, V):
        if V.element().geometric_dimension() == 2 and statistics:
            raise TypeError("No sampling of turbulence statistics in 2D")

        if len(self.dims) == 2:
            x = self.create_dense_grid()
            if V.element().geometric_dimension() == 2:
                x = x[:, :2]
            self.probes = (StatisticsProbes(x.flatten(), V, V.num_sub_spaces()==0) if statistics else
                           Probes(x.flatten(), V))
        else:
            # Add plane by plane to avoid excessive memory use
            x = self.create_xy_slice(0)
            self.probes = (StatisticsProbes(x.flatten(), V, V.num_sub_spaces()==0) if statistics else
                           Probes(x.flatten(), V)) 
            for i in range(1, len(self.dz[2])):
                x = self.create_xy_slice(i)
                args = (x.flatten(), V, V.num_sub_spaces()==0) if statistics else (x.flatten(), V)
                self.probes.add_positions(*args)
                 
    def __call__(self, *args):
        self.probes(*args)
        
    def __getitem__(self, i):
        return self.probes[i]
    
    def __iter__(self): 
        self.i = 0
        return self

    def next(self):
        try:
            p = self[self.i]
        except:
            raise StopIteration
        self.i += 1
        return p   
        
    def read_grid(self, restart):
        if restart.endswith(".vtk"):
            statistics = self.read_grid_from_vtk(restart)
        else:
            statistics = self.read_grid_from_hdf5(restart)
        return statistics
    
    def set_data_from_file(self, filename='', tstep=0):
        if filename.endswith('.vtk'):
            self.set_data_from_vtk(filename)
        else:
            self.set_data_from_hdf5(filename=filename, tstep=tstep)

    def create_coordinate_vectors(self):
        """Create the vectors that span the entire computational box.
        
        dx, dy, dz are lists of length dim (2 for 2D, 3 for 3D)
        Each item of the list is an array spanning the box/slice in
        that direction. For example, a 3D UnitCube of size 5 x 4 x 3
        with tangents corresponding to the xyz-axes can be represented 
        as
        
        dx = [array([ 0., 0.25, 0.5, 0.75, 1.]),
              array([ 0., 0., 0., 0.]),
              array([ 0., 0., 0.])]

        dy = [array([ 0., 0., 0., 0., 0.]),
              array([ 0., 0.33333, 0.66667, 1.]),
              array([ 0., 0., 0.])]
 
        dz = [array([ 0., 0., 0., 0., 0.]),
              array([ 0., 0., 0., 0.]),
              array([ 0., 0.5, 1.])]
        
        We store the entire vector in each direction because each
        vector can be modified and need not be uniform (like here).
        
        """
        dX, dL = self.dX, self.dL
        dims = array(self.dims)
        for i, N in enumerate(dims):
            dX[i, :] = dX[i, :] / numpy_norm(dX[i, :]) * dL[i] / N
        
        dx, dy, dz = [], [], []
        for k in range(len(dims)):
            dx.append(linspace(0, dX[k][0]*dims[k], dims[k]))
            dy.append(linspace(0, dX[k][1]*dims[k], dims[k]))
            dz.append(linspace(0, dX[k][2]*dims[k], dims[k]))
                
        dx, dy, dz = self.modify_mesh(dx, dy, dz)
        return dx, dy, dz

    def modify_mesh(self, dx, dy, dz):
        """Use this function to for example skew a grid towards a wall.
        
        If you are looking at a channel flow with -1 < y < 1 and 
        walls located at +-1, then you can skew the mesh using e.g.
        
            dy[1][:] = arctan(pi*(dy[1][:]+origin[1]))/arctan(pi)-origin[1]
            
        Note: origin[1] will be -1 and 0 <= dy <= 2
        
        """
        return dx, dy, dz
    
    def create_dense_grid(self):
        """Create a dense 2D slice or 3D box."""
        origin = self.origin
        dx, dy, dz = self.dx, self.dy, self.dz
        dims = array(self.dims)
        dim = dims.prod()
        x = zeros((dim, 3))
        if len(dims) == 3:
            x[:, 0] = repeat(dx[2], dims[0]*dims[1])[:] + resize(repeat(dx[1], dims[0]), dim)[:] + resize(dx[0], dim)[:] + origin[0]
            x[:, 1] = repeat(dy[2], dims[0]*dims[1])[:] + resize(repeat(dy[1], dims[0]), dim)[:] + resize(dy[0], dim)[:] + origin[1]
            x[:, 2] = repeat(dz[2], dims[0]*dims[1])[:] + resize(repeat(dz[1], dims[0]), dim)[:] + resize(dz[0], dim)[:] + origin[2]
        else:
            x[:, 0] = repeat(dx[1], dims[0])[:] + resize(dx[0], dim)[:] + origin[0]
            x[:, 1] = repeat(dy[1], dims[0])[:] + resize(dy[0], dim)[:] + origin[1]
            x[:, 2] = repeat(dz[1], dims[0])[:] + resize(dz[0], dim)[:] + origin[2]
        return x

    def create_xy_slice(self, nz):
        origin = self.origin
        dx, dy, dz = self.dx, self.dy, self.dz
        dims = array(self.dims)
        dim = dims[0]*dims[1]
        x = zeros((dim, 3))
        x[:, 0] = resize(repeat(dx[1], dims[0]), dim)[:] + resize(dx[0], dim)[:] + origin[0] + dx[2][nz]
        x[:, 1] = resize(repeat(dy[1], dims[0]), dim)[:] + resize(dy[0], dim)[:] + origin[1] + dy[2][nz]
        x[:, 2] = resize(repeat(dz[1], dims[0]), dim)[:] + resize(dz[0], dim)[:] + origin[2] + dz[2][nz]
        return x
        
    def read_grid_from_hdf5(self, filename="restart.h5"):
        """Read grid data from filename"""
        f = h5py.File(filename, 'r', driver='mpio', comm=comm)
        self.origin = f.attrs['origin']
        self.dL = f.attrs['dL']
        self._num_eval = f.attrs['num_evals']
        dx0 = f.attrs['dx0']
        dx1 = f.attrs['dx1']
        dx2 = f.attrs['dx2']
        self.dx = [dx0[0], dx1[0], dx2[0]]
        self.dy = [dx0[1], dx1[1], dx2[1]]
        self.dz = [dx0[2], dx1[2], dx2[2]]
        self.dims = map(len , self.dx)
        statistics = "stats" in f["FEniCS"]
        f.close()
        return statistics
        
    def set_data_from_hdf5(self, filename="restart.h5", tstep=None):
        """Set data in probes using values stored in filename.
        If no specific tstep is provided for regular probe, then choose 
        the tstep with the highest number.
        """
        f = h5py.File(filename, 'r', driver='mpio', comm=comm)
        ids = self.probes.get_probe_ids()        
        nn = len(ids)
        
        if "stats" in f['FEniCS'].keys():
            loc = 'FEniCS/stats'
            rs = ["UMEAN", "VMEAN", "WMEAN", "uu", "vv", "ww", "uv", "uw", "vw"]
            data = zeros((nn, 9))
            for i, name in enumerate(rs):
                data[:, i] = f[loc][name].value.flatten()[ids]   
            self.probes.set_probes_from_ids(data.flatten(), self._num_eval)
        else:            
            if tstep == None:
                step = f['FEniCS'].keys()
                step.sort()
                loc = 'FEniCS/' + step[-1]
            else:
                loc = 'FEniCS/tstep'+str(tstep)
            data = zeros((nn, len(f[loc])))
            if len(f[loc]) == 1:
                data[:, 0] = f[loc]['Scalar'].value.flatten()[ids]
            else:
                for ii in range(self.probes.value_size()):
                    data[:, ii] = f[loc]['Comp-{}'.format(ii)].value.flatten()[ids]
                    
            self.probes.set_probes_from_ids(data.flatten())
        f.close()
        
    def get_ijk(self, global_index):
        """return i, j, k indices of structured grid based on global index"""
        d = self.dims
        return (global_index % d[0], 
               (global_index % (d[0]*d[1])) / d[0], 
                global_index / (d[0]*d[1]))
        
    def toh5(self, N, tstep, filename="restart.h5"):
        """Dump probes to HDF5 file. The data can be used for 3D visualization
        using voluviz or to restart the probe. 
        """
        f = h5py.File(filename, 'w', driver='mpio', comm=comm)
        d = self.dims
        if not 'origin' in f.attrs:
            f.attrs.create('origin', self.origin)
            for i in range(len(self.dx)):
                f.attrs.create('dx'+str(i), array([self.dx[i], self.dy[i], self.dz[i]]))
            f.attrs.create('dL', self.dL)
        if not 'num_evals' in f.attrs:
            f.attrs.create('num_evals', self.probes.number_of_evaluations())
        else:
            f.attrs['num_evals'] = self.probes.number_of_evaluations()
        try:
            f.create_group('FEniCS')
        except ValueError:
            pass
        if type(self.probes) == StatisticsProbes and N == 0:
            loc = 'FEniCS/stats'
        else:
            loc = 'FEniCS/tstep'+str(tstep)
        try:
            f.create_group(loc)
        except ValueError:
            pass
        
        # Create datasets if not there already
        d = list(d)
        if len(d) == 2: 
            d.append(1)
        dimT = (d[2], d[1], d[0])
        if self.probes.value_size() == 1:
            try:
                f.create_dataset(loc+"/Scalar", shape=dimT, dtype='f')
            except RuntimeError:
                print 'RuntimeError'
        elif type(self.probes) != StatisticsProbes:
            try:
                for ii in range(self.probes.value_size()):
                    f.create_dataset(loc+"/Comp-{}".format(ii), shape=dimT, dtype='f')
            except:
                pass
            
        elif type(self.probes) == StatisticsProbes:
            if N == 0:
                try:
                    num_evals = self.probes.number_of_evaluations()
                    f.create_dataset(loc+"/UMEAN", shape=dimT, dtype='f')
                    f.create_dataset(loc+"/VMEAN", shape=dimT, dtype='f')
                    f.create_dataset(loc+"/WMEAN", shape=dimT, dtype='f')
                    rs = ["uu", "vv", "ww", "uv", "uw", "vw"]
                    for i in range(3, 9):
                        f.create_dataset(loc+"/"+rs[i-3], shape=dimT, dtype='f')
                except RuntimeError:
                    pass                    
            else: # Just dump latest snapshot
                try:                        
                    f.create_dataset(loc+"/U", shape=dimT, dtype='f')
                    f.create_dataset(loc+"/V", shape=dimT, dtype='f')
                    f.create_dataset(loc+"/W", shape=dimT, dtype='f')
                except RuntimeError:
                    pass
                
        # Last dimension of box is shared amongst processors
        # In case d[2] % Nc is not zero the last planes are distributed
        # between the processors starting with the highest rank and then 
        # gradually lower
        
        comm.barrier()
        Nc = comm.Get_size()
        myrank = comm.Get_rank()
        Np = self.probes.get_total_number_probes()
        planes_per_proc = d[2] / Nc
        # Distribute remaining planes 
        if Nc-myrank <= (d[2] % Nc):
            planes_per_proc += 1
            
        # Let all processes know how many planes the different processors own
        all_planes_per_proc = comm.allgather(planes_per_proc)
        cum_last_id = cumsum(all_planes_per_proc)
        owned_planes = zeros(Nc+1, 'I')
        owned_planes[1:] = cum_last_id[:]
                            
        # Store owned data in z0
        z0 = zeros((d[0], d[1], planes_per_proc, self.probes.value_size()), dtype=float32)
        zhere = zeros(self.probes.value_size(), dtype=float32)
        zrecv = zeros(self.probes.value_size(), dtype=float32)
        sendto = zeros(Nc, 'I')
        # Run through all probes and send them to the processes 
        # that owns the plane its at
        for global_index, probe in self.probes:
            i, j, k = self.get_ijk(global_index)
            owner = argmax(cum_last_id > k)
            zhere[:] = probe.get_probe_at_snapshot(N)
            if owner != myrank:
                # Send data to owner
                sendto[owner] +=1
                comm.send(global_index, dest=owner, tag=101)
                comm.Send(zhere, dest=owner, tag=102)
            else:
                # myrank owns the current probe and can simply store it
                z0[i, j, k-owned_planes[myrank], :] = zhere[:]
        # Let all processors know who they are receiving data from
        recvfrom = zeros((Nc, Nc), 'I')
        comm.Allgather(sendto, recvfrom)
        # Receive the data
        for ii in range(Nc):
            num_recv = recvfrom[ii, myrank]
            for kk in range(num_recv):
                global_index = comm.recv(source=ii, tag=101)
                i, j, k = self.get_ijk(global_index)
                comm.Recv(zrecv, source=ii, tag=102)
                z0[i, j, k-owned_planes[myrank], :] = zrecv[:]
                
        # Voluviz has weird ordering so transpose some axes
        z0 = z0.transpose((2,1,0,3))
        # Write owned data to hdf5 file
        owned = slice(owned_planes[myrank], owned_planes[myrank+1])
        comm.barrier()
        if owned.stop > owned.start:
            if self.probes.value_size() == 1:
                f[loc+"/Scalar"][owned, :, :] = z0[:, :, :, 0]
            elif type(self.probes) != StatisticsProbes:
                for ii in range(self.probes.value_size()):
                    f[loc+"/Comp-{}".format(ii)][owned, :, :] = z0[:, :, :, ii]
            elif type(self.probes) == StatisticsProbes:
                if N == 0:
                    num_evals = self.probes.number_of_evaluations()
                    f[loc+"/UMEAN"][owned, :, :] = z0[:, :, :, 0] / num_evals
                    f[loc+"/VMEAN"][owned, :, :] = z0[:, :, :, 1] / num_evals
                    f[loc+"/WMEAN"][owned, :, :] = z0[:, :, :, 2] / num_evals
                    rs = ["uu", "vv", "ww", "uv", "uw", "vw"]
                    for ii in range(3, 9):
                        f[loc+"/"+rs[ii-3]][owned, :, :] = z0[:, :, :, ii] / num_evals
                else: # Just dump latest snapshot
                    f[loc+"/U"][owned, :, :] = z0[:, :, :, 0]
                    f[loc+"/V"][owned, :, :] = z0[:, :, :, 1]
                    f[loc+"/W"][owned, :, :] = z0[:, :, :, 2]
        comm.barrier()
        f.close()

    def surf(self, N, component=0):
        """surf plot of scalar or one component of tensor"""
        if comm.Get_size() > 1:
            print "No surf for multiple processors"
            return
        if len(self.dims) == 3:
            print "No surf for 3D cube"
            return
        z = self.array(N=N, component=component).reshape(*self.dims[::-1])
        x = self.create_dense_grid()
        surfc(x[:, 0].reshape(*self.dims[::-1]), x[:, 1].reshape(*self.dims[::-1]), z, indexing='xy')

    def array(self, N=None, filename=None, component=None, root=0):
        """Dump data to numpy format on root processor for all or one snapshot"""
        return self.probes.array(N=N, filename=filename, component=component, root=root)
    
    def tovtk(self, N, filename):
        """Dump probes to VTK file."""
        is_root = comm.Get_rank() == 0
        z = self.array(N=N)
        if is_root:
            d = self.dims
            d = (d[0], d[1], d[2]) if len(d) > 2 else (d[0], d[1], 1)
            grid = pyvtk.StructuredGrid(d, self.create_dense_grid())
            v = pyvtk.VtkData(grid, "Probe data. Evaluations = {}".format(self.probes.number_of_evaluations()))
            if self.probes.value_size() == 1:
                v.point_data.append(pyvtk.Scalars(z, name="Scalar"))
            elif self.probes.value_size() == 3:
                v.point_data.append(pyvtk.Vectors(z, name="Vector"))
            elif self.probes.value_size() == 9: # StatisticsProbes
                if N == 0:
                    num_evals = self.probes.number_of_evaluations()
                    v.point_data.append(pyvtk.Vectors(z[:, :3]/num_evals, name="UMEAN"))
                    rs = ["uu", "vv", "ww", "uv", "uw", "vw"]
                    for i in range(3, 9):
                        v.point_data.append(pyvtk.Scalars(z[:, i]/num_evals, name=rs[i-3]))
                else: # Just dump latest snapshot
                    v.point_data.append(pyvtk.Vectors(z[:, :3], name="U"))
            else:
                raise TypeError("Only vector or scalar data supported for VTK")
            v.tofile(filename)
                
    def average(self, i):
        """Contract results by averaging along axis. Useful for homogeneous
        turbulence geometries like channels or cylinders"""
        z = self.probes.array()
        if comm.Get_rank() == 0:
            z = reshape(z, self.dims[::-1] + [z.shape[-1]]).transpose((2,1,0,3))
            if isinstance(i, int):
                return z.mean(i)
            else:
                if len(i) == 2:
                    assert(i[1] > i[0])
                elif len(i) == 3:
                    assert(i[0] == 0 and i[1] == 1 and i[2] == 2)
                for k, j in enumerate(i):
                    j -= k
                    z = z.mean(j)
                return z
            
    def read_grid_from_vtk(self, filename="restart.vtk"):
        """Read vtk-file stored previously with tovtk."""
        p = pyvtk.VtkData(filename)
        xn = array(p.structure.points)
        dims = p.structure.dimensions
        try:
            N = eval(p.header.split(" ")[-1])
        except:
            N = 0
        num_evals = N if isinstance(N, int) else 0
                        
        self.dims = list(dims)
        d = copy.deepcopy(self.dims)
        d.reverse()
        x = squeeze(reshape(xn, d + [3]))            
        if 1 in self.dims: self.dims.remove(1)
        
        if len(squeeze(array(self.dims))) == 2:
            xs = [x[0,:,0] - x[0,0,0]]
            xs.append(x[:,0,0] - x[0,0,0])
            ys = [x[0,:,1] - x[0,0,1]]
            ys.append(x[:,0,1] - x[0,0,1])
            zs = [x[0,:,2] - x[0,0,2]]
            zs.append(x[:,0,2] - x[0,0,2])
        else:
            xs = [x[0,0,:,0] - x[0,0,0,0]]
            xs.append(x[0,:,0,0] - x[0,0,0,0])
            xs.append(x[:,0,0,0] - x[0,0,0,0])
            ys = [x[0,0,:,1] - x[0,0,0,1]]
            ys.append(x[0,:,0,1] - x[0,0,0,1])
            ys.append(x[:,0,0,1] - x[0,0,0,1])
            zs = [x[0,0,:,2] - x[0,0,0,2]]
            zs.append(x[0,:,0,2] - x[0,0,0,2])
            zs.append(x[:,0,0,2] - x[0,0,0,2])
            
        self.dx, self.dy, self.dz = xs, ys, zs
        self.origin = xn[0]
        self._num_eval = num_evals
        return True
        
    def set_data_from_vtk(self, filename):        
        p = pyvtk.VtkData(filename)
        vtkdata = p.point_data.data

        ids = self.probes.get_probe_ids()        
        nn = len(ids)

        # Count the number of fields
        i = 0
        for d in vtkdata:
            if hasattr(d, 'vectors'):
                i += 3
            else:
                i += 1        

        # Put all field in data
        data = zeros((array(self.dims).prod(), i))
        i = 0
        for d in vtkdata:
            if hasattr(d, 'vectors'):
                data[:, i:(i+3)] = array(d.vectors)
                i += 3
            else:
                data[:, i] = array(d.scalars)
                i += 1
                
        self.probes.restart_probes(data.flatten(), self._num_eval)

    def arithmetic_mean(self, N=0, component=None):
        z = self.array(N=N, component=component)
        a = 0.0
        if comm.Get_rank() == 0:
            a = sum(z) / count_nonzero(z)
        a = comm.bcast(a, root=0)
        return a
                    
# Create a class for a skewed channel mesh 
class ChannelGrid(StructuredGrid):
    def modify_mesh(self, dx, dy, dz):
        """Create grid skewed towards the walls located at y = 1 and y = -1"""
        dy[1][:] = cos(pi*(dy[1][:]+self.origin[1] - 1.) / 2.) - self.origin[1]
        return dx, dy, dz

def interpolate_nonmatching_mesh_python(u0, V):
    """interpolate from Function u0 in V0 to a Function in V."""
    V0 = u0.function_space()
    mesh1 = V.mesh()
    gdim = mesh1.geometry().dim()
    owner_range = V.dofmap().ownership_range()
    u = Function(V)
    xs = V.dofmap().tabulate_all_coordinates(mesh1).reshape((-1, gdim))
    
    def extract_dof_component_map(dof_component_map, VV, comp):
        
        if VV.num_sub_spaces() == 0:
            ss = VV.dofmap().collapse(VV.mesh())
            comp[0] += 1 
            for val in ss[1].values():
                dof_component_map[val] = comp[0]
            
        else:
            for i in range(VV.num_sub_spaces()):
                Vs = VV.extract_sub_space(array([i], 'I'))
                extract_dof_component_map(dof_component_map, Vs, comp)

    # Create a map from global dof to component in Mixed space
    dof_component_map = {}
    comp = array([-1], 'I')
    extract_dof_component_map(dof_component_map, V, comp)
    
    # Now find the values of all these locations using Probes
    # Do this sequentially to save memory    
    dd = zeros(u.vector().local_size())
    for i in range(MPI.num_processes()):
        # Put all locations of this mesh on other processes
        xb = comm.bcast(xs, root=i)        
        # Probe these locations and store result in u
        probes = Probes(xb.flatten(), V0)
        probes(u0)  # evaluate the probes
        data = probes.array(N=0, root=i)
        probes.clear()
        if i == comm.Get_rank():
            if V.num_sub_spaces() < 1:
                u.vector().set_local(squeeze(data))
            else:
                for j in range(data.shape[0]):
                    dof = j + owner_range[0]
                    dd[j] = data[j, dof_component_map[dof]]
                u.vector().set_local(dd)
        
    return u

    
# Test the probe functions:
if __name__=='__main__':
    import time
    #set_log_active(False)
    set_log_level(20)

    mesh = UnitCubeMesh(16, 16, 16)
    #mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    Vv = VectorFunctionSpace(mesh, 'CG', 1)
    W = V * Vv
    
    # Just create some random data to be used for probing
    x0 = interpolate(Expression('x[0]'), V)
    y0 = interpolate(Expression('x[1]'), V)
    z0 = interpolate(Expression('x[2]'), V)
    s0 = interpolate(Expression('exp(-(pow(x[0]-0.5, 2)+ pow(x[1]-0.5, 2) + pow(x[2]-0.5, 2)))'), V)
    v0 = interpolate(Expression(('x[0]', '2*x[1]', '3*x[2]')), Vv)
    w0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]')), W)
        
    x0.update()
    y0.update()
    z0.update()
    s0.update()
    v0.update()
    w0.update()
    
    x = array([[1.5, 0.5, 0.5], [0.2, 0.3, 0.4], [0.8, 0.9, 1.0]])
    p = Probes(x.flatten(), W)
    x = x*0.9 
    p.add_positions(x.flatten(), W)
    for i in range(6):
        p(w0)
        
    print p.array(2, "testarray")         # dump snapshot 2
    print p.array(filename="testarray")   # dump all snapshots
    print p.dump("testarray")

    # Test StructuredGrid
    # 3D box
    origin = [0.25, 0.25, 0.25]               # origin of box
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # coordinate vectors (scaled in StructuredGrid)
    dL = [0.5, 0.5, 0.5]                      # extent of slice in both directions
    N  = [9, 9, 3]                           # number of points in each direction
    
    ## 2D slice
    #origin = [-0.5, -0.5, 0.5]               # origin of slice
    #vectors = [[1, 0, 0], [0, 1, 0]]    # directional tangent directions (scaled in StructuredGrid)
    #dL = [2., 2.]                      # extent of slice in both directions
    #N  = [50, 50]                           # number of points in each direction
   
    # Test scalar first
    sl = StructuredGrid(V, N, origin, vectors, dL)
    sl(s0)     # probe once
    sl(s0)     # probe once more
    sl.surf(0) # check first probe
    sl.tovtk(0, filename="dump_scalar.vtk")
    sl.toh5(0, 1, 'dump_scalar.h5')
   # sl.toh5(0, 2, 'dump_scalar.h5')
   # sl.toh5(0, 3, 'dump_scalar.h5')
    print sl.arithmetic_mean()
    
    # then vector
    sl2 = StructuredGrid(Vv, N, origin, vectors, dL)
    for i in range(5): 
        sl2(v0)     # probe a few times
    sl2.surf(3)     # Check the fourth probe instance
    sl2.tovtk(3, filename="dump_vector.vtk")
    sl2.toh5(0, 1, 'dump_vector.h5')
    #sl2.toh5(0, 2, 'dump_vector.h5')
    #sl2.toh5(0, 3, 'dump_vector.h5')
    
    # Test statistics
    sl3 = StructuredGrid(V, N, origin, vectors, dL, True)
    for i in range(10): 
        sl3(x0, y0, z0)     # probe a few times
        #sl3(v0)
    sl3.surf(0)     # Check 
    sl3.tovtk(0, filename="dump_mean_vector.vtk")
    sl3.tovtk(1, filename="dump_latest_snapshot_vector.vtk")
    sl3.toh5(0, 1, 'reslowmem.h5')    
        
    # Restart probes from sl3
    sl4 = StructuredGrid(V, restart='reslowmem.h5')
    for i in range(10): 
        sl4(x0, y0, z0)     # probe a few more times    
    sl4.tovtk(0, filename="dump_mean_vector_restart_h5.vtk")
    sl4.toh5(0, 0, 'restart_reslowmem.h5')
    
    # Try mixed space
    sl5 = StructuredGrid(W, N, origin, vectors, dL)
    sl5(w0)     # probe once
    sl5(w0)     # probe once more
    sl5.toh5(0, 1, 'dump_mixed.h5')
    sl6 = StructuredGrid(W, restart='dump_mixed.h5')
    sl6.toh5(0, 1, 'dump_mixed_again.h5')
    
    ## 3D box
    #tol = 1e-8
    #origin = [0.+tol, -1.+tol, -1.+tol]                     # origin of box
    #vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # coordinate vectors (scaled in StructuredGrid)
    #dL = [4.-2*tol, 2.-2*tol, 2.-2*tol]                           # extent of slice in both directions
    #N  = [10, 10, 10]                           # number of points in each direction
    #mesh = BoxMesh(0., -1., -1., 4., 1., 1., 10, 10, 10)
    #V = FunctionSpace(mesh, 'CG', 1)
    #x0 = interpolate(Expression('x[0]'), V)
    #y0 = interpolate(Expression('x[1]'), V)
    #z0 = interpolate(Expression('x[2]'), V)
    #slc = ChannelGrid(V, N, origin, vectors, dL, True)
    #slc(x0, y0, z0)
    #slc.tovtk(0, 'testing_dump.vtk')
    #slc.toh5(0, 1, 'testing_dump.h5')
    
    
    #WS = W * W
    ##w11 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]', 'x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]')), WS)
    #w11 = interpolate_nonmatching_mesh(Expression(('x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]', 'x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]')), WS)
    #WS2 = W2 * W2
    #w11.update()
    ##x1 = interpolate_nonmatching_mesh(w11, WS2)

    ##ff = File('test_project_nonmatching.pvd')
    ##ff << u
    ##plot(u)
    #plot(w11[0], title='mixed')
    #list_timings()
    
    interactive()
    
