import os
import sys
import subprocess
import re
import shutil
import importlib

valgrind_models = [
                    '2ph_comp',
                    'Chem_benchmark_new',
                    'GeoRising'
                   ]

# if the user passed extra models, append them (comma-separated):
extra = os.environ.get('APPEND_VALGRIND_MODEL')
if extra:
    for m in extra.split(','):
        m = m.strip()
        if m and m not in valgrind_models:
            valgrind_models.append(m)

log_folder = os.path.join('models', '_valgrind_logs')

# valgrind suppression file (adjust path as needed)
suppression_file = 'helper_scripts/valgrind-python.supp'

def run_model(model, days):
    # model path
    model_path = os.path.join('models', model)
    if not os.path.isdir(model_path):
        print(f"[SKIP] Model directory not found: {model_path}")
        return False

    # step into the model directory
    cwd = os.getcwd()
    os.chdir(model_path)

    # add it also to system path to load modules
    sys.path.insert(0, os.path.abspath(r'.'))

    # clear any cached bytecode
    shutil.rmtree("__pycache__", ignore_errors=True)

    # import the model definition and run
    try:
        # assume model.py defines a class Model
        mod = importlib.import_module('model')
        m = mod.Model()
        m.init()
        m.run(days)
        print(f"[OK] {model}")
        success = True
    except Exception as e:
        print(f"[FAIL] {model} → {e}")
        success = False
    finally:
        # restore working directory
        os.chdir(cwd)

    return 0 if success else 1

# patterns to extract leak info and errors from Valgrind log
MSG_PATTERNS = {
    'definitely_lost':    r'definitely lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'indirectly_lost':    r'indirectly lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'possibly_lost':      r'possibly lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'still_reachable':    r'still reachable:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'error_summary':      r'ERROR SUMMARY:\s+(\d+)'  # total errors
}

def analyze_log(log_path):
    """Parse Valgrind log and return a dict of summary values."""
    summary = {}
    try:
        content = open(log_path, 'r').read()
    except FileNotFoundError:
        return {'error': 'Log file not found'}

    ret_code = 0
    for key, pattern in MSG_PATTERNS.items():
        m = re.search(pattern, content)
        if m:
            if key == 'error_summary':
                ret_code = int(m.group(1))
                summary[key] = ret_code
            else:
                summary[key] = { 'bytes': m.group(1), 'blocks': m.group(2) }
    return summary, ret_code

def run_valgrind_for_model(model):
    # file paths
    vg_log = os.path.join(log_folder, f'{model}.vg.log')
    prog_out = os.path.join(log_folder, f'{model}_mainpy.log')
    prog_err = os.path.join(log_folder, f'{model}_mainpy_err.log')
    summary_file = os.path.join(log_folder, f'{model}.summary.txt')

    # Build the inline Python snippet to invoke run_model_direct
    days = 1
    py_snippet = (
        'import sys, os; '
        'sys.path.insert(0, os.getcwd()); '
        'from helper_scripts.valgrind_profile import run_model; '
        f'sys.exit(0 if run_model(model="{model}", days={days}) else 1)'
    )

    # valgrind command profiling run_model_direct
    cmd = [
        'valgrind',
        '--quiet',
        '--error-exitcode=1',
        f'--log-file={vg_log}',
        '--',
        'python', '-c', py_snippet
    ]

    print(f'Running Valgrind for model {model}...')

    # set environment: disable pymalloc so Valgrind sees everything
    env = os.environ.copy()
    env['PYTHONMALLOC'] = 'malloc'
    darts_path = subprocess.check_output([
        'python', '-c',
        'import os, darts; print(os.path.dirname(darts.__file__))'
    ]).decode().strip()
    env['LD_LIBRARY_PATH'] = f"{darts_path}:{env.get('LD_LIBRARY_PATH','')}"

    # run
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, timeout=1800
    )

    with open(prog_out, 'w') as out_f, open(prog_err, 'w') as err_f:
        try:
            proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, env=env, timeout=1800)
            # proc = subprocess.run(cmd, shell=True, env=env, timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"ERROR: timeout profiling model {model}")
            return

    # analyze and write summary
    summary, ret_code = analyze_log(vg_log)
    with open(summary_file, 'w') as sf:
        sf.write(f'Valgrind summary for model: {model}\n')
        for k, v in summary.items():
            if isinstance(v, dict):
                sf.write(f'  {k}: {v["bytes"]} bytes in {v["blocks"]} blocks\n')
            else:
                sf.write(f'  {k}: {v}\n')

    if ret_code != 0:
        print(f'Valgrind detected errors for model {model}, exit code {ret_code}.')
    elif proc.returncode != 0:
        print(f'Model {model} returned exit code {proc.returncode}.')
    else:
        print(f'Valgrind completed successfully for model {model}.')

def main():
    # check for valgrind availability
    if not shutil.which('valgrind'):
        print('Error: valgrind not found in PATH')
        sys.exit(1)

    # log folder
    os.makedirs(log_folder, exist_ok=True)

    # run models
    for model in valgrind_models:
        run_valgrind_for_model(model)

if __name__ == '__main__':
    main()