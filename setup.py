from setuptools import setup, find_packages

setup(
    name="speed_profile_generator",
    version="0.1.0",
    python_requires=">=3.12",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["*.webapp.frontend*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "h5py",
        "scipy",
        "matplotlib",
        "tabulate",
    ],
    extras_require={
        "test": ["pytest"],
        "webapp": ["fastapi>=0.100.0", "uvicorn>=0.23.0", "pydantic>=2.0.0"],
        "report": ["reportlab"],
    },
)
