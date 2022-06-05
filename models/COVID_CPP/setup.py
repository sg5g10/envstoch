from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools
import pybind11

class get_pybind_include(object):

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

cpp_args = ["-std=c++11","-Wall", "-Wextra","-DNDEBUG", "-O3"]

ext_modules = [
    Extension(
    'seeiir_ode',
        ['seeiir_ode.cpp'],
        include_dirs=[get_pybind_include(), 
        get_pybind_include(user=True)],
    language='c++',
    extra_compile_args = cpp_args,
    ),
    Extension(
    'death_lik',
        ['likelihoods.cpp'],
        include_dirs=[get_pybind_include(), 
        get_pybind_include(user=True)],
    language='c++',
    extra_compile_args = cpp_args,
    )    
]

setup(
    name='SEEIIR System',
    version='0.0.1',
    author='sanmitra ghosh',
    author_email='sanmitra.ghosh@mrc-bsu.cam.ac.uk',
    description='SEEIIR_ODE',
    ext_modules=ext_modules,
)
