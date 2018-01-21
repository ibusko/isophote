from setuptools import setup

setup(
    name="isophote",
    url="https://github.com/ibusko/isophote",
    packages=["ellipse"],
# Doesn't work because of relative paths to test/data in test/test_*.py:
#    test_suite="test",
)
