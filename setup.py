from setuptools import setup, find_packages

setup(
    name='cognomic',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click>=8.0.0',
        'aiohttp>=3.8.0',
        'openai>=0.27.0',
        'networkx>=2.8.0',
        'pyyaml>=6.0.0',
    ],
    entry_points={
        'console_scripts': [
            'cognomic=cognomic.cli:run',
        ],
    },
)
