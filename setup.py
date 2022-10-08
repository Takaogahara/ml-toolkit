#!/usr/bin/env python
import setuptools
from ml_toolkit import __version__

from distutils.command.clean import clean
from distutils.command.install import install


def get_requirements_list(file):
    requirements = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            requirements.append(line)
    return requirements


class MyInstall(install):
    def run(self):
        install.run(self)
        c = clean(self.distribution)
        c.all = True
        c.finalize_options()
        c.run()


requirements = get_requirements_list("requirements.txt")

setuptools.setup(name="ml-toolkit",
                 version=__version__,
                 description="Machine Learning Toolkit",
                 author="Rafael Lopes Almeida",
                 author_email="fael.rlopes@gmail.com",
                 url="https://github.com/Takaogahara/ml-toolkit",
                 packages=setuptools.find_packages(),
                 entry_points={"console_scripts": [
                     "gnn_toolkit = gnn_toolkit.main:toolkit_parser"]},
                 install_requires=requirements,
                 cmdclass={'install': MyInstall})
