python setup.py clean
python setup.py build bdist_wheel
pip install --upgrade --no-deps --force-reinstall dist/darts-0.1.0-cp36-cp36m-win_amd64.whl