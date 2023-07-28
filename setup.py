from setuptools import setup
__lib_name__ = "stGCL"
__lib_version__ = "1.0.2"
__description__ = "Deciphering spatial domains from spatially resolved transcriptomics with adaptive graph attention auto-encoder"
__author__ = "Daoliang Zhang"
__author_email__ = "201720386@mail.edu.sdu.cn"
__license__ = "MIT"
__keywords__ = ["spatial transcriptomics", "Deep learning", "Graph attention auto-encoder"]
__requires__ = ["requests",]

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    __author__="Daoliang Zhang",
    __email__ = "201720386@mail.edu.sdu.cn",
    license = __license__,
    packages = ["stGCL"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = __long_description__
)
