import os

from setuptools import setup

if __name__ == "__main__":
    # update the version number
    version = open("VERSION", "r").read().strip()

    cwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cwd, "toybnb", "__version__.py"), "w") as f:
        f.write(f"__version__ = '{version}'\n")

    setup(
        name="toybnb",
        version=version,
        description="""A toy BnB solver for MILP""",
        long_description=open("README.md", "rt").read(),
        long_description_content_type="text/markdown",
        license="MIT",
        packages=[
            "toybnb",
            "toybnb.scip",
            "toybnb.scip.ecole",
            "toybnb.scip.ecole.il",
        ],
        python_requires=">=3.9",
        install_requires=[
            "numpy",
            "networkx",
            "scipy",
            "pyscipopt",
            "ecole",
        ],
        test_requires=[
            "pytest",
        ],
    )
