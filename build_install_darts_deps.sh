#!/bin/bash

python3 setup.py clean
python3 setup.py build bdist_wheel
pip3 install darts-package/dist/*.whl
