import setuptools
import numpy
import sys
import os

if sys.platform == "win32":
    compile_args = ["/openmp", "-O3"]
    link_args = ["/openmp"]
elif sys.platform == "darwin":
    compile_args = ["-O3"]
    link_args = []
else:
    compile_args = ["-fopenmp", "-O3"]
    link_args = ["-fopenmp"]

setuptools.setup(
    name="kurDISTANCE",
    packages=setuptools.find_packages(),
    include_dirs=[numpy.get_include()],
    ext_modules=[
        setuptools.Extension(
            "kurDISTANCE.lock_step",
            sources=["src/lock_step_cmodule.c", 
                "src/math_utils.c"],
            extra_compile_args=compile_args,
            extra_link_args=link_args
        ),
        setuptools.Extension(
            "kurDISTANCE.elastic",
            sources=["src/elastic_cmodule.c", 
                "src/math_utils.c"],
            extra_compile_args=compile_args,
            extra_link_args=link_args
        ),
        setuptools.Extension(
            "kurDISTANCE.trend_based",
            sources=["src/trend_based_cmodule.c", 
                "src/math_utils.c"],
            extra_compile_args=compile_args,
            extra_link_args=link_args
        )
    ]
)

