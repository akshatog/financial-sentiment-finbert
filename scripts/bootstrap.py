"""Create a local environment and install CPU dependencies reproducibly."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev', action='store_true', help='Include test dependencies.')
    args = parser.parse_args()
    environment = ROOT / '.venv'
    executable = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not executable.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    child_env = os.environ.copy()
    child_env['PIP_CACHE_DIR'] = str(ROOT / '.cache' / 'pip')
    subprocess.run([str(executable), '-m', 'pip', 'install', '--upgrade', 'pip'], cwd=ROOT, env=child_env, check=True)
    requirements = 'requirements-dev.txt' if args.dev else 'requirements.txt'
    subprocess.run([str(executable), '-m', 'pip', 'install', '-r', requirements], cwd=ROOT, env=child_env, check=True)
    print(f'Environment ready: {executable}')


if __name__ == '__main__':
    main()
