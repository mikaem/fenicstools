__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-01-08"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import cppimport

compiled_module = cppimport.imp('fenicstools.fem.common')

def getMemoryUsage(rss=True):
    return compiled_module.getMemoryUsage(rss)

def SetMatrixValue(A, val):
    compiled_module.SetMatrixValue(A, val)

