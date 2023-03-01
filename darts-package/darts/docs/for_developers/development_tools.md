# Development tools

## Build and installation instructions

## Testing

Run the tests locally as follows:

```bash
cd darts-models # Go to folder containing the tests
python run_test_suite2.py
```

## Documentation

The documentation uses [Sphinx](https://www.sphinx-doc.org/en/master/) and supports both markdown (.md) and reStructuredText (.rst). To generate the documentation locally you will have to first install some specific dependencies for the documentation:

```bash
cd darts-package/darts/docs # go to the folder containing the documentation
pip install -r requirements.txt
```

Then you can compile the documentation:

```bash
make clean
make html # html can be changed for pdf epub and othe formats supported by sphinx
```

## Software release
