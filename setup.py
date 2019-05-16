#!/usr/bin/env python

from distutils.core import setup

# Version number
major = 2019
minor = 1

setup(name = "fenicstools",
      version = "%d.%d" % (major, minor),
      description = "fenicstools -- tools for postprocessing in FEniCS programs",
      author = "Mikael Mortensen",
      author_email = "mikaem@math.uio.no",
      url = 'https://github.com/mikaem/fenicstools.git',
      classifiers = [
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages = ["fenicstools"],
      package_dir = {"fenicstools": "fenicstools"},
      package_data = {"fenicstools": ["probe/*.h",
                                      "probe/*.cpp",
                                      "fem/*.cpp",
                                      "dofmapplotter/*.py",
                                      "dofmapplotter/cpp/*.cpp"]},
    )
