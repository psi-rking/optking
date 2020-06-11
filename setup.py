import setuptools
import versioneer

short_description = "OptKing is a python geometry optimization module originally written for PSI by R. A. King."

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = short_description

if __name__ == "__main__":
    setuptools.setup(
        name='OptKing',
        description='A geometry optimizer for quantum chemistry.',
        author='Rollin King',
        author_email='rking@bethel.edu',
        url="https://github.com/psi-rking/optking",
        license='BSD-3C',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.7',
            'qcelemental>=0.12.0',
            'qcengine>=0.12.0',
        ],
        extras_require={
            'docs': [
                'sphinx',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
        },
        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=False,
        long_description=long_description,
        long_description_content_type="text/markdown"
    )
