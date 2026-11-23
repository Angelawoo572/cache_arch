# Source this on Sacramento before building or running gem5 for this study.
# Everything it points to lives under the cache root.
export CACHE_ROOT=/home/qianruw/cache
export GEM5_ROOT="$CACHE_ROOT/external/gem5"
export GEM5_STUDY="$CACHE_ROOT/gem5_neural_prefetch"
export GEM5_BIN="$GEM5_ROOT/build/X86/gem5.opt"
export TMPDIR="$CACHE_ROOT/.runtime/tmp"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/.runtime/pycache"
export XDG_CACHE_HOME="$CACHE_ROOT/.runtime/xdg"
export PIP_CACHE_DIR="$CACHE_ROOT/.runtime/pip-cache"
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
# gem5 reads PYTHON_CONFIG from the environment. The default python3-config is Python 3.6 here.
export PYTHON_CONFIG=/usr/bin/python3.11-config
source /opt/rh/gcc-toolset-13/enable
source "$CACHE_ROOT/.runtime/gem5-venv/bin/activate"
