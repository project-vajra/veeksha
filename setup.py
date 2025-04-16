from setuptools import setup, find_packages

setup(
    name="veeksha",
    version="0.1.0",
    packages=find_packages(include=["veeksha", "veeksha.*"]),
    include_package_data=True,
)
