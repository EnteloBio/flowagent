from setuptools import setup, find_packages

setup(
    name='cognomic',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add your project's dependencies here, e.g.
        # 'flask',
    ],
    entry_points={
        'console_scripts': [
            'cognomic-server=cognomic.api.server:main',  # Corrected module path
            'cognomic-run=cognomic.cli:run',  # New entry point for running workflows
        ],
    },
)
