from setuptools import find_packages, setup

REQUIRED_PACKAGES = ["gcsfs>=2021.4.0"]

setup(
    name="trainer",
    version="0.5",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Simple training application",
)