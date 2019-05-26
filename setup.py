#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

req_file = "requirements.txt"

def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements(req_file)

setup(name='model',
      version='0.1',
      description='A Framework for auto medical image detection.',
      url='https://github.com/funhere/med/auto-medical-detection',
      packages=find_packages(exclude=['test', 'test.*']),
      install_requires=install_reqs,
      dependency_links=[],
      )
