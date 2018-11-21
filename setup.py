from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='text_summarizer',
    version='0.1',
    url='https://github.com/lambdaofgod/text-summarizer',
    author='Jakub Bartczuk',
    packages=find_packages(),
    install_requires=requirements
)