from setuptools import find_packages, setup

setup(
    name="flowagent",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "flowagent=flowagent.cli:run",
        ],
    },
)
