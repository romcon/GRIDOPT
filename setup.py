# *****************************************************#
# This file is part of GRIDOPT.                       #
#                                                     #
# Copyright (c) 2015, Tomas Tinoco De Rubira.         #
#                                                     #
# GRIDOPT is released under the BSD 2-clause license. #
# *****************************************************#

import os
import sys
from setuptools import setup, find_packages
import py_compile
from setuptools.command.build_py import build_py
from setuptools.command.bdist_egg import bdist_egg
from wheel.bdist_wheel import bdist_wheel


exec(open(os.path.join('gridopt', 'version.py')).read())

# Custom distribution build commands
class bdist_wheel_compiled(bdist_wheel):
    """Small customizations to build compiled only wheel."""
    description = 'build compiled wheel distribution'


class bdist_egg_compiled(bdist_egg):
    """Small customizations to build compiled only egg."""
    description = 'build compiled egg distribution'


if len(sys.argv) > 1 and 'compiled' in sys.argv[1]:

    class build_py(build_py):
        """
        A custom build_py command to exclude source files from packaging and
        include compiled pyc files instead.
        """
        def byte_compile(self, files):
            for file in files:
                full_path = os.path.abspath(file)
                if file.endswith('.py'):
                    print("{}  compiling and unlinking".format(file))
                    py_compile.compile(file, cfile=file+'c')
                    os.unlink(file)
                elif file.endswith('pyx') or file.endswith('pxd'):
                    print("{}  unlinking".format(file))
                    os.unlink(file)

    extra_cmd_classes = {'bdist_wheel_compiled': bdist_wheel_compiled,
                         'bdist_egg_compiled': bdist_egg_compiled,
                         'build_py': build_py}

else:
    extra_cmd_classes = {'bdist_wheel_compiled': bdist_wheel_compiled,
                         'bdist_egg_compiled': bdist_egg_compiled}


setup(name='GRIDOPT',
      zip_safe=False,
      version=__version__,
      description='Power Grid Optimization Library',
      url='https://github.com/romcon/GRIDOPT',
      author='Adam Wigington, Fan Zhang, Swaroop Guggilam',
      author_email='awigington@epri.com',
      license='BSD 2-Clause License',
      cmdclass=extra_cmd_classes,
      packages=find_packages(),
      entry_points={'console_scripts': ['gridopt=gridopt.scripts.gridopt:main']},
      classifiers=['Development Status :: 5 - Production/Stable',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10'],
      install_requires=['cython>=0.20.1',
                        'numpy>=1.11.2',
                        'scipy>=0.18.1',
                        'pfnet>=1.3.6',
                        'optalg>=1.1.9',
                        'pytest'])
