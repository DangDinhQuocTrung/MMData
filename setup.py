#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import re
from setuptools import setup, Command


__PATH__ = os.path.abspath(os.path.dirname(__file__))


def read_readme():
    with open(os.path.join(__PATH__, "README.md")) as f:
        return f.read()
    pass


def read_version():
    with open(os.path.join(__PATH__, "mmdata/__init__.py")) as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find __version__ string")


def read_requirements():
    with open(os.path.join(__PATH__, "requirements.txt")) as f:
        return f.read().split("\n")[:-1]
    pass


__version__ = read_version()


class DeployCommand(Command):
    description = "Build and deploy the package to PyPI."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    @staticmethod
    def status(s):
        print(s)

    def run(self):
        assert "dev" not in __version__, (
            "Only non-devel versions are allowed. "
            "__version__ == {}".format(__version__))

        try:
            from shutil import rmtree
            self.status("Removing previous builds ...")
            rmtree(os.path.join(__PATH__, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution ...")
        os.system("{0} setup.py sdist".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine ...")
        ret = os.system("twine upload dist/*")
        if ret != 0:
            sys.exit(ret)

        self.status("Creating git tags ...")
        os.system("git tag v{0}".format(__version__))
        os.system("git tag --list")
        sys.exit()


setup_requires = []

install_requires = read_requirements()

tests_requires = [
    "mockito>=1.2.1",
]
if sys.version_info >= (3, 5):
    tests_requires += ["pytest>=5.4.1"]
    setup_requires += ["pytest-runner>=5.0"],
else:
    tests_requires += ["pytest<5.0", "more_itertools<8.0"]
    setup_requires += ["pytest-runner<5.0"],

setup(
    name="mmdata",
    version=__version__,
    license="MIT",
    description="A utility tool pose an MMD model and generate massive training data for 3D mesh reconstruction.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Cinnamon/gpucop",
    author="DangDinhQuocTrung",
    author_email="ddqtrung@gmail.com",
    keywords="mmd mesh pmx vmd reconstruction",
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
    ],
    packages=["mmdata"],
    install_requires=install_requires,
    extras_require={"test": tests_requires},
    setup_requires=setup_requires,
    tests_require=tests_requires,
    entry_points={
        "console_scripts": ["gpucop=gpucop:main"],
    },
    cmdclass={
        "deploy": DeployCommand,
    },
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.5",
)
