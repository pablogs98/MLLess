from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import lithops
import Cython
import re
import os
import shutil
import subprocess


def copy_bin(path, dir_="."):
    filenames = os.listdir(dir_)
    for file in filenames:
        if re.match(r".*cpython-38.*\.so", file):
            shutil.copy(f"{dir_}/{file}", f"{path}/{file}")


try:
    os.environ['PYTHONPATH'] += os.path.abspath(os.curdir)
except KeyError:
    os.environ['PYTHONPATH'] = os.path.abspath(os.curdir)

# Compile on linux (IBM Cloud functions lithops runtime)
subprocess.call(['sh', os.path.abspath('lithops_runtime/compile_linux_cython.sh')],
                cwd=os.path.abspath("./lithops_runtime"))

Cython.Compiler.Options.annotate = False
lithops_path = '{}libs'.format(lithops.__file__).replace("__init__.py", "")
includes = [np.get_include()]
macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

shutil.copyfile("MLLess/models/model.pxd", f"{lithops_path}/model.pxd")

setup(name='MLLess',
      ext_modules=cythonize(
          [Extension("lithops_runtime.model", sources=["MLLess/models/model.pyx"],
                     include_dirs=includes,
                     define_macros=macros),

           Extension("lithops_runtime.mf_model", sources=["MLLess/models/mf_model.pyx"],
                     include_dirs=includes,
                     define_macros=macros),

           Extension("lithops_runtime.lr_model", sources=["MLLess/models/lr_model.pyx"],
                     include_dirs=includes,
                     define_macros=macros),

           Extension("lithops_runtime.sparse_lr_model",
                     sources=["MLLess/models/sparse_lr_model.pyx"],
                     include_dirs=includes,
                     define_macros=macros)
           ], compiler_directives={'language_level': "3"}, annotate=True))

copy_bin(lithops_path, "./lithops_runtime")
shutil.rmtree('build')
