from setuptools import setup, find_packages

setup(
    use_scm_version=True,
    packages=find_packages(include=["iita_python*"]),
)