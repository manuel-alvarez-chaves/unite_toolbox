from setuptools import setup, find_packages

setup(
    name="unite_toolbox",
    version="0.1",
    contact_email="malvarez062@gmail.com",
    packages=find_packages(),
    install_requires=["numpy >= 1.25.0", "scipy >= 1.10.1"],
)