from setuptools import find_packages, setup

setup(
    name="cognomic",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "cognomic=cognomic.cli:run",
        ],
    },
)
