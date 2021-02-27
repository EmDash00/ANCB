import setuptools  # type: ignore

with open("README.md", "r") as fh:
    for _ in range(4):
        fh.readline()
    long_description_text = fh.read()

with open("requirements.txt") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="ancb",
    version="0.1.1",
    author="Drason Chow",
    author_email="drasonchow.business@gmail.com",
    description="Fast, efficient, and powerful NumPy "
                "compatible circular buffers",
    long_description=long_description_text,
    long_description_content_type="text/markdown",
    url="https://github.com/EmDash00/ANCB",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
