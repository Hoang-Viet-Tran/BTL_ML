python3.13 -m venv .venv
ECHO "Virtual environment created."
source .venv/bin/activate
#!/usr/bin/env bash
set -euo pipefail

# init.sh - create and activate a Python venv, install dependencies
# Usage: PYTHON=python3.13 ./scripts/init.sh

green() {
	printf "\033[1;32m%s\033[0m\n" "$1"
}

# Allow override, default to python3.13
: "${PYTHON:=python3.13}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
	if command -v python3 >/dev/null 2>&1; then
		PYTHON=python3
	elif command -v python >/dev/null 2>&1; then
		PYTHON=python
	else
		printf "No suitable python interpreter found (tried python3.13, python3, python)\n" >&2
		exit 1
	fi
fi

green "Creating virtual environment with $PYTHON..."
$PYTHON -m venv .venv
green "Virtual environment created."

# shellcheck source=/dev/null
source .venv/bin/activate
green "Virtual environment activated."

green "Installing dependencies..."
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
	python -m pip install -r requirements.txt
else
	printf "requirements.txt not found in %s\n" "$(pwd)" >&2
fi
green "Dependencies installed."

green "Installing ipykernel..."
python -m pip install ipykernel
green "ipykernel installed."

green "Setup complete."