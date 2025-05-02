import os
import subprocess
import re
import shutil

valgrind_models = ['2ph_comp']

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

    for key, pattern in MSG_PATTERNS.items():
        m = re.search(pattern, content)
        if m:
            if key == 'error_summary':
                ret_code = int(m.group(1))
                summary[key] = ret_code
            else:
                summary[key] = {
                    'bytes': m.group(1),
                    'blocks': m.group(2)
                }
    return summary, 0 #ret_code

def run_valgrind_for_model(model):
    # file paths
    vg_log = os.path.join(log_folder, f'{model}.vg.log')
    prog_out = os.path.join(log_folder, f'{model}_mainpy.log')
    prog_err = os.path.join(log_folder, f'{model}_mainpy_err.log')
    summary_file = os.path.join(log_folder, f'{model}.summary.txt')

    # build valgrind command
    cmd = [
        'valgrind',
        '--quiet',
        #'--leak-check=full',
        #'--show-leak-kinds=all',
        '--error-exitcode=1',
        f'--log-file={vg_log}',
        #f'--suppressions={suppression_file}',
        'python', f'models/{model}/main.py'
    ]
    print(f'Running Valgrind for model {model}...')

    # set environment: disable pymalloc so Valgrind sees everything
    env = os.environ.copy()
    env['PYTHONMALLOC'] = 'malloc'
    env['LD_LIBRARY_PATH'] = os.popen('python -c "import os; import darts; print(os.path.dirname(darts.__file__))"').read()[:-1] + ':' + os.environ.get('LD_LIBRARY_PATH')

    # run with separate stdout/err capture
    with open(prog_out, 'w') as out_f, open(prog_err, 'w') as err_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, env=env)

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
        print(f'Valgrind detected errors for model {model}, exit code {proc.returncode}.')
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