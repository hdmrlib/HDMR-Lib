# PyPI Release Checklist

This document outlines the steps to release HDMR-Lib on PyPI.

## Pre-Release Checklist

### 1. Code Quality
- [x] All core functionality implemented (HDMR, EMPR)
- [x] All backends working (NumPy, PyTorch, TensorFlow)
- [x] CUDA/GPU support added
- [x] Metrics module implemented (MSE, sensitivity analysis)
- [x] All placeholder functions removed or implemented

### 2. Tests
- [x] Test suite created (test_hdmr.py, test_empr.py, test_backends.py, test_metrics.py)
- [ ] Run all tests: `pytest tests/ -v`
- [ ] All tests passing
- [ ] Test coverage >80%

### 3. Documentation
- [x] README.md complete with examples
- [x] Examples folder with working scripts
- [x] Docstrings in all public functions
- [x] Installation instructions
- [x] GPU/CUDA documentation

### 4. Package Files
- [x] setup.py created
- [x] pyproject.toml created
- [x] LICENSE file (MIT)
- [x] MANIFEST.in
- [x] .gitignore
- [x] requirements.txt
- [x] __version__ in __init__.py

### 5. Metadata
- [ ] Update author name in setup.py
- [ ] Update author email in setup.py
- [ ] Update GitHub URL in setup.py
- [ ] Update repository URL in pyproject.toml
- [ ] Choose package name (check availability on PyPI)

## Release Steps

### Step 1: Update Metadata
```bash
# Edit setup.py and pyproject.toml
# Update:
# - author
# - author_email  
# - url (GitHub repository)
# - version (if needed)
```

### Step 2: Test Installation Locally
```bash
# Install in development mode
pip install -e .

# Test imports
python -c "from hdmr import HDMR; from empr import EMPR; print('✓ Imports working')"

# Run examples
python examples/basic_usage.py

# Run tests
pytest tests/ -v
```

### Step 3: Build Distribution
```bash
# Install build tools
pip install build twine

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build package
python -m build
```

This creates:
- `dist/hdmr-lib-0.1.0.tar.gz` (source distribution)
- `dist/hdmr_lib-0.1.0-py3-none-any.whl` (wheel)

### Step 4: Test on TestPyPI (Recommended)
```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ hdmr-lib

# Verify it works
python -c "import hdmr_lib; print(hdmr_lib.__version__)"
```

### Step 5: Upload to PyPI
```bash
# Upload to real PyPI
twine upload dist/*

# Test installation
pip install hdmr-lib

# Verify
python -c "import hdmr_lib; print(hdmr_lib.__version__)"
```

### Step 6: Tag Release on GitHub
```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

## Post-Release

1. Create GitHub Release with changelog
2. Announce on relevant forums/communities
3. Update documentation website (if any)
4. Monitor PyPI download statistics
5. Respond to issues and pull requests

## Troubleshooting

### Package Name Already Exists
If `hdmr-lib` is taken, try:
- `hdmr-pro`
- `hdmr-toolkit`
- `pyhdmr`
- Check availability: https://pypi.org/project/YOUR-PACKAGE-NAME/

### Import Errors After Installation
Make sure package structure is correct:
```
hdmr-lib/
├── hdmr.py
├── empr.py
├── metrics.py
├── backends/
│   └── __init__.py
└── __init__.py
```

### Tests Failing
```bash
# Run specific test
pytest tests/test_hdmr.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Version History

- **0.1.0** (Initial Release)
  - HDMR and EMPR decomposition
  - Multi-backend support (NumPy, PyTorch, TensorFlow)
  - CUDA/GPU acceleration
  - Sensitivity analysis
  - Comprehensive test suite
  - Example scripts

## Resources

- PyPI Documentation: https://packaging.python.org/
- TestPyPI: https://test.pypi.org/
- Twine: https://twine.readthedocs.io/
- Setuptools: https://setuptools.pypa.io/

