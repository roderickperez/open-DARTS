import darts.engines as darts_engines
from darts.engines import print_build_info as engines_pbi
from darts.print_build_info import print_build_info as package_pbi
from darts.engines import sim_params

import time
import sys, os, shutil
import subprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def get_model_dirs():
    main_files = []
    for root, dirs, files in os.walk("."):
        path = root.split(os.sep)
        for file in files:
            if file =='main.py':
                main_files.append(root)
    return main_files

def run_models(model_dirs):
    # print build info
    engines_pbi()

    # number of threads for the parallel runs
    os.environ['OMP_NUM_THREADS'] = '10'

    overwrite = '0'
    if os.getenv('UPLOAD_PKL') == '1':
        overwrite = '1'

    n_failed = n_total = 0
    failed_list = []

    init_dir = os.getcwd()
    # check main.py files runs, without comparison of pkl files
    n_failed_mainpy = n_total_mainpy = 0
    for mdir in model_dirs:
        print('running main.py for model', mdir)
        n_total_mainpy += 1
        os.chdir(os.path.join(init_dir, mdir))
        import subprocess
        log_dir = os.path.join(*[init_dir, '_logs', mdir])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        starting_time = time.time()
        mrun = subprocess.run(["python", "main.py"],
                              stdout=open(os.path.join(log_dir, 'out.log'), 'w'),
                              stderr=open(os.path.join(log_dir, 'err.log'), 'w'))
        ending_time = time.time()
        rcode = mrun.returncode
        n_failed_mainpy += int(rcode != 0)
        if rcode == 0:
            print(mdir, 'OK, \t%.2f s' % (ending_time - starting_time))
        else:
            with open(os.path.join(log_dir, 'err.log'), 'r') as f:
                s = f.readlines()
            print(mdir, 'FAIL, \t%.2f s' % (ending_time - starting_time), 'Msg:', s[-1])
            failed_list.append(mdir)

    os.chdir(init_dir)
    n_failed += n_failed_mainpy
    n_total += n_total_mainpy

    n_passed = n_total - n_failed
    print("Passed", n_passed, "of", n_total, "models. ")
    return n_failed, n_total, failed_list

def notebooks_clear_out_dir():
    dir_out = 'out'
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.mkdir(dir_out)

def notebooks_list():
    notebooks = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith('.ipynb'):
                if '.ipynb_checkpoints' not in dirs and 'checkpoint' not in file:
                    notebooks.append(root + os.sep + file)
    return notebooks

def run_notebook(notebook_fname):
    '''
    run notebook from given filename
    return 0 if success, 1 if failed
    '''
    notebook_fname_out = os.path.join('out', notebook_fname)
    os.makedirs(os.path.dirname(notebook_fname_out), exist_ok=True)

    with open(notebook_fname) as f:
        nb = nbformat.read(f, as_version=4)

    # initialize python3 notebook processor with timeout 5 min
    ep = ExecutePreprocessor(timeout=300, kernel_name='python3')

    root_dir = os.getcwd()

    # run the notebook file
    ret = 0
    try:
        os.chdir(os.path.dirname(notebook_fname))
        ep.preprocess(nb, {'metadata': {'path': '.'}})
    except CellExecutionError:
        msg = 'Error executing the notebook "%s".\n' % notebook_fname
        msg += 'See notebook "%s" for the traceback.' % notebook_fname_out
        print(msg)
        ret = 1
    finally:
        os.chdir(root_dir)
        # write result notebook
        with open(notebook_fname_out, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    return ret


def run_notebooks(notebooks):
    '''
    runs all ipynb files in the current directory
    returns the number of failed notebooks (0 if all succeed)
    '''

    n_total = n_failed = 0
    failed_list = []

    for filename in notebooks:
        starting_time = time.time()
        rcode = run_notebook(filename)
        ending_time = time.time()
        n_failed += rcode
        n_total += 1
        if rcode == 0:
            print(filename, 'OK, \t%.2f s' % (ending_time - starting_time))
        else:
            print(filename, 'FAIL, \t%.2f s' % (ending_time - starting_time))
            failed_list.append(filename)

    print('PASSED', n_total - n_failed, 'of', n_total, 'notebooks')
    return n_failed, n_total, failed_list

if __name__ == '__main__':
    n_failed = 0

    notebooks_clear_out_dir()
    notebooks = notebooks_list()
    n_failed_n, n_total_n, failed_list_n = run_notebooks(notebooks)
    n_failed += n_failed_n

    print('Failed notebooks:\n\t', '\n\t'.join(failed_list_n))

    model_dirs = get_model_dirs()
    n_failed_m, n_total_m, failed_list_m = run_models(model_dirs)
    n_failed += n_failed_m

    print('Failed notebooks:\n\t', '\n\t'.join(failed_list_n))
    print('Failed models   :\n\t', '\n\t'.join(failed_list_m))

    print('PASSED', n_total_n - n_failed_n, 'of', n_total_n, 'notebooks')
    print('PASSED', n_total_m - n_failed_m, 'of', n_total_m, 'models')
    
    # exit with code equal to number of failed models and notebooks
    exit(n_failed)

