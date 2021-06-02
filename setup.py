import setuptools

__version__ = '0.0.1dev1'

def _parse_requirements(path):
    with open(path) as f:
        return [
            line.rstrip()
            for line in f
            if not (line.isspace() or line.startswith('#'))
        ]

requirements = _parse_requirements('requirements.txt')

from setuptools.command.install import install
from subprocess import getoutput

class PostInstall(install):
    pkgs = ' git+https://github.com/open-mmlab/mmdetection.git#v2.13.0'
    def run(self):
        install.run(self)
        print(getoutput('pip install'+self.pkgs))


setuptools.setup(
    name='ozeu',
    version=__version__,
    packages=setuptools.find_packages(),
    install_requires=requirements,
    cmdclass={'install': PostInstall}
)
