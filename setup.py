import os
import re
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

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = re.sub(
        "<!--- long-description-skip-begin -->.*<!--- long-description-skip-end -->",
        "",
        readme_file.read(),
        flags=re.S | re.M,
    )

setup(
    name="qiskit-toqm",
    version="0.1.0",
    description="Qiskit transpiler passes for the TOQM algorithm",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/qiskit-toqm/qiskit-toqm",
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
    keywords="qiskit sdk quantum toqm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/qiskit_toqm/native",
    install_requires=REQUIREMENTS,
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    python_requires=">=3.7",
    project_urls={
        "Bug Tracker": "https://github.com/qiskit-toqm/qiskit-toqm/issues",
        "Documentation": "https://github.com/qiskit-toqm/qiskit-toqm",
        "Source Code": "https://github.com/qiskit-toqm/qiskit-toqm",
    },
    entry_points={
        'qiskit.transpiler.routing': [
            'toqm = qiskit_toqm.toqm_plugin:ToqmSwapPlugin',
        ],
    },
)
