import setuptools
import numpy

setuptools.setup(
    name="kurDISTANCE",
    packages=setuptools.find_packages(),
    include_dirs=[numpy.get_include()],
    ext_modules=[
        setuptools.Extension(
            "kurDISTANCE.lock_step",
            sources=["src/lock_step_cmodule.c"]
        ),
        setuptools.Extension(
            "kurDISTANCE.elastic",
            sources=["src/elastic_cmodule.c"]
        ),
        setuptools.Extension(
            "kurDISTANCE.trend_based",
            sources=["src/trend_based_cmodule.c"]
        )
    ]
)
