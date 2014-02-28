__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
This module contains functionality for efficiently probing a Function many times. 
"""
from dolfin import *
from numpy import zeros, array, squeeze, reshape 
import os, inspect
from mpi4py.MPI import COMM_WORLD as comm

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
        return self.get_probe_id(i), Probe(self.get_probe(i))

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
        return self.get_probe_id(i), StatisticsProbe(self.get_probe(i))

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

