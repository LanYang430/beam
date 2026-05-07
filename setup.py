from setuptools import setup, find_packages

setup(
    name="beam",
    version="0.1.0",
    author="Lan Yang",
    author_email="lyang430@gatech.edu",
    description="Boosted Enhanced sampling through Machine-learned collective variables",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LanYang430/beam.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "mdtraj>=1.9.0",
        "deeptime>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "jupyter>=1.0.0",
        ],
    },
)
