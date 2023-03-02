python setup.py clean
python setup.py build bdist_wheel
pip install --upgrade --no-deps --force-reinstall dist/open_darts-0.1.1-cp36-cp36m-win_amd64.whl