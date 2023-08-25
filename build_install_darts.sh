#!/bin/bash

python3 setup.py clean
python3 setup.py build bdist_wheel
python3 -m pip install .
