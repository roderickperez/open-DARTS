"""
Valgrind (Memcheck) helper for darts-flash.

Purpose
-------
Run selected pytest targets, Python scripts, or arbitrary commands under
Valgrind with child-process tracing and suppression files, then summarize
the results into human-readable reports.

Key behaviors
-------------
- Sets PYTHONMALLOC=malloc for accurate Python allocator visibility.
- Traces child processes (e.g., subprocesses spawned by pytest or scripts).
- Automatically uses common suppression files if found in this repo.
- Produces four files per run in --log-dir (default: valgrind_logs/):
  <name>.vg.log, <name>.out.log, <name>.err.log, <name>.summary.txt
- Exits with code 1 if Valgrind reports errors, or the command fails.

Usage examples
--------------
- Run a single pytest file (quiet):
    python3 helper_scripts/valgrind_check.py pytest -q tests/python/test_flash_vlaq.py

- Run an entire directory:
    python3 helper_scripts/valgrind_check.py pytest tests/python

- Run a specific test node:
    python3 helper_scripts/valgrind_check.py pytest \
        tests/python/test_flash_vlaq.py::TestClass::test_case

- Run a Python script with arguments:
    python3 helper_scripts/valgrind_check.py python path/to/script.py -- --arg value

- Run an arbitrary command:
    python3 helper_scripts/valgrind_check.py cmd -- \
        python3 -m pytest -q tests/python/test_flash_vlaq.py

Useful flags
------------
- --log-dir DIR     Directory to store Valgrind logs (default: valgrind_logs)
- --timeout SEC     Per-run timeout in seconds
- --omp-threads N   OMP_NUM_THREADS value (default: 1)
- --suppressions F  Additional suppression files
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time


DEFAULT_LOG_DIR = os.path.join('valgrind_logs')

# Patterns to extract leak info and errors from Valgrind log
MSG_PATTERNS = {
    'definitely_lost': r'definitely lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'indirectly_lost': r'indirectly lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'possibly_lost': r'possibly lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'still_reachable': r'still reachable:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'error_summary': r'ERROR SUMMARY:\s+(\d+)',  # total errors
}


def _default_suppressions() -> list[str]:
    """Return a list of default suppression files if present in this repo."""
    candidates = [
        os.path.join('thirdparty', 'pybind11', 'tests', 'valgrind-python.supp'),
        os.path.join('thirdparty', 'pybind11', 'tests', 'valgrind-numpy-scipy.supp'),
        os.path.join('helper_scripts', 'valgrind-python.supp'),
    ]
    return [p for p in candidates if os.path.isfile(p)]


def _sanitize_filename(text: str) -> str:
    """Return a filesystem-safe name for logs based on a human-readable label."""
    base = text.strip().replace(' ', '_')
    base = base.replace(os.sep, '_')
    base = base.replace('/', '_')
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', base)


def analyze_log(log_path: str) -> tuple[list[dict], int]:
    """Parse Valgrind log and return list of summary dicts plus accumulated error count."""
    try:
        content = open(log_path, 'r', encoding='utf-8', errors='ignore').read()
    except FileNotFoundError:
        return [], 1

    blocks = re.split(r"(?m)==\d+==\s+HEAP SUMMARY:", content)
    summary_list: list[dict] = []
    total_errors = 0

    for block in blocks[1:]:
        block = 'HEAP SUMMARY:' + block
        report: dict = {}
        pid_match = re.search(r'==([0-9]+)==', block)
        report['pid'] = pid_match.group(1) if pid_match else None
        for key, pattern in MSG_PATTERNS.items():
            match = re.search(pattern, block)
            if match:
                if key == 'error_summary':
                    err = int(match.group(1))
                    report[key] = err
                    total_errors += err
                else:
                    report[key] = {'bytes': match.group(1), 'blocks': match.group(2)}
        summary_list.append(report)

    return summary_list, total_errors


def _compute_ld_library_path(env: dict) -> None:
    """Attempt to prepend dartsflash module directory to LD_LIBRARY_PATH."""
    try:
        out = subprocess.check_output(
            [sys.executable, '-c', 'import os, importlib.util; m=importlib.util.find_spec("dartsflash"); print(os.path.dirname(m.origin) if m and m.origin else "")']
        ).decode().strip()
        if out:
            env['LD_LIBRARY_PATH'] = f"{out}:{env.get('LD_LIBRARY_PATH', '')}"
    except Exception:
        pass


def _valgrind_base_cmd(log_file: str, suppressions: list[str]) -> list[str]:
    """Construct the base Valgrind memcheck command with common options."""
    cmd = [
        'valgrind',
        '--tool=memcheck',
        '--leak-check=full',
        '--show-leak-kinds=definite,indirect',
        '--track-origins=yes',
        '--read-var-info=yes',
        '--trace-children=yes',
        '--error-exitcode=0',  # do not fail immediately; we'll parse logs
        f'--log-file={log_file}',
    ]
    for s in suppressions:
        cmd.append(f'--suppressions={s}')
    # Generate suppressions interactively when new noise appears
    cmd.append('--gen-suppressions=all')
    cmd.append('--')
    return cmd


def run_single_valgrind(run_cmd: list[str], log_dir: str, name: str, timeout: int | None, omp_threads: int, suppressions: list[str]) -> bool:
    """Run a single command under Valgrind and write logs and a summary.

    Returns True on success (no Valgrind errors and zero exit status), False otherwise.
    """
    os.makedirs(log_dir, exist_ok=True)
    safe_name = _sanitize_filename(name)
    vg_log = os.path.join(log_dir, f'{safe_name}.vg.log')
    prog_out = os.path.join(log_dir, f'{safe_name}.out.log')
    prog_err = os.path.join(log_dir, f'{safe_name}.err.log')
    summary_file = os.path.join(log_dir, f'{safe_name}.summary.txt')

    cmd = _valgrind_base_cmd(vg_log, suppressions) + run_cmd

    env = os.environ.copy()
    env['PYTHONMALLOC'] = 'malloc'
    env['OMP_NUM_THREADS'] = str(omp_threads)
    _compute_ld_library_path(env)

    print(f'Running under Valgrind: {name}')
    started = time.time()

    with open(prog_out, 'w') as out_f, open(prog_err, 'w') as err_f:
        try:
            proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f'ERROR: timeout while running {name}')
            return False

    summary_list, total_errors = analyze_log(vg_log)
    with open(summary_file, 'w') as sf:
        sf.write(f'Valgrind summary for: {name}\n')
        for report in summary_list:
            pid = report.get('pid', 'unknown')
            sf.write(f"\nReport for PID: {pid}\n")
            for k, v in report.items():
                if k == 'pid':
                    continue
                if isinstance(v, dict):
                    sf.write(f"  {k}: {v['bytes']} bytes in {v['blocks']} blocks\n")
                else:
                    sf.write(f"  {k}: {v}\n")
        sf.write(f"\nAccumulated ERROR SUMMARY: {total_errors}\n")

    elapsed = time.time() - started
    if proc.returncode != 0:
        print(f'[FAIL] Command exited with {proc.returncode}\t{elapsed:.2f} s')
        return False
    if total_errors != 0:
        print(f'[FAIL] Valgrind detected {total_errors} errors\t{elapsed:.2f} s')
        return False

    print(f'[OK] No Valgrind errors\t{elapsed:.2f} s')
    return True


def main() -> int:
    """CLI entry point: parse arguments and dispatch to run modes."""
    parser = argparse.ArgumentParser(
        description='Run Valgrind on pytest targets or Python commands with suppressions and child tracing.'
    )
    parser.add_argument('--log-dir', default=DEFAULT_LOG_DIR, help='Directory for Valgrind logs (default: %(default)s)')
    parser.add_argument('--timeout', type=int, default=None, help='Per-run timeout in seconds')
    parser.add_argument('--omp-threads', type=int, default=1, help='OMP_NUM_THREADS value (default: 1)')
    parser.add_argument(
        '--suppressions',
        action='append',
        default=[],
        metavar='FILE',
        help='Additional suppression files (repeatable)',
    )

    subparsers = parser.add_subparsers(dest='mode', required=True)

    p_pytest = subparsers.add_parser('pytest', help='Run pytest targets under Valgrind')
    p_pytest.add_argument('targets', nargs='+', help='Test files, dirs or node ids')
    p_pytest.add_argument('-q', '--quiet', action='store_true', help='Pass -q to pytest')

    p_python = subparsers.add_parser('python', help='Run a Python script under Valgrind')
    p_python.add_argument('script', help='Path to Python script')
    p_python.add_argument('script_args', nargs=argparse.REMAINDER, help='Arguments for the script')

    p_cmd = subparsers.add_parser('cmd', help='Run an arbitrary command under Valgrind')
    p_cmd.add_argument('command', nargs=argparse.REMAINDER, help='Command to execute (after --)')

    args = parser.parse_args()

    suppressions = _default_suppressions()
    if args.suppressions:
        for sup in args.suppressions:
            if os.path.isfile(sup):
                suppressions.append(sup)
            else:
                print(f'Warning: suppression file not found: {sup}')

    if not shutil.which('valgrind'):
        print('Error: valgrind not found in PATH')
        return 1

    all_ok = True
    if args.mode == 'pytest':
        # Expand any directories into per-file pytest targets so we continue
        # running other files even if one fails.
        expanded_targets: list[str] = []
        for t in args.targets:
            if os.path.isdir(t):
                for root, _dirs, files in os.walk(t):
                    for fn in files:
                        if (fn.startswith('test_') or fn.endswith('_test.py')) and fn.endswith('.py'):
                            expanded_targets.append(os.path.join(root, fn))
            else:
                expanded_targets.append(t)

        # De-duplicate and maintain stable order
        seen = set()
        deduped = []
        for t in expanded_targets:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        for target in deduped:
            run_cmd = [sys.executable, '-m', 'pytest']
            if args.quiet:
                run_cmd.append('-q')
            run_cmd.append(target)
            ok = run_single_valgrind(
                run_cmd=run_cmd,
                log_dir=args.log_dir,
                name=f'pytest_{target}',
                timeout=args.timeout,
                omp_threads=args.omp_threads,
                suppressions=suppressions,
            )
            all_ok = all_ok and ok
    elif args.mode == 'python':
        run_cmd = [sys.executable, args.script] + (args.script_args or [])
        all_ok = run_single_valgrind(
            run_cmd=run_cmd,
            log_dir=args.log_dir,
            name=f'python_{args.script}',
            timeout=args.timeout,
            omp_threads=args.omp_threads,
            suppressions=suppressions,
        )
    elif args.mode == 'cmd':
        if not args.command:
            print('Error: no command provided. Use: cmd -- <your command>')
            return 1
        all_ok = run_single_valgrind(
            run_cmd=args.command,
            log_dir=args.log_dir,
            name='cmd',
            timeout=args.timeout,
            omp_threads=args.omp_threads,
            suppressions=suppressions,
        )

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())


