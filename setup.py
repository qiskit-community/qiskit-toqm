import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

setup(
    name="qiskit-toqm",
    version="0.0.2",
    description="Qiskit transpiler passes for the TOQM algorithm",
    author="Qiskit Development Team",
    author_email="hello@qiskit.org",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/qiskit_toqm/native",
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    python_requires=">=3.7",
    project_urls={
        "Bug Tracker": "https://github.com/kevinhartman/qiskit-toqm/issues",
        "Documentation": "https://github.com/kevinhartman/qiskit-toqm",
        "Source Code": "https://github.com/kevinhartman/qiskit-toqm",
    },
)
