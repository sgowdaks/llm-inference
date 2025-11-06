# Contributing to Qwen ONNX Inference

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/llm-inference.git
cd llm-inference
git submodule update --init --recursive
```

### 2. Set Up Development Environment

Follow the instructions in [BUILD.md](BUILD.md) to set up your development environment.

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

## Development Workflow

### 1. Make Changes

- Write clean, readable code
- Follow the existing code style
- Add comments for complex logic
- Keep commits focused and atomic

### 2. Code Style

#### C++
- Use C++17 standard features
- Follow RAII principles
- Use smart pointers (avoid raw pointers)
- Prefer `const` and `constexpr` where applicable
- Use meaningful variable names

```cpp
// Good
const std::vector<int32_t> token_ids = tokenizer->encode(prompt);

// Avoid
auto v = t->enc(p);
```

#### Python
- Follow PEP 8 style guide
- Use type hints
- Write docstrings for functions and classes
- Maximum line length: 100 characters

```python
def run_inference(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 1.0
) -> str:
    """Run ONNX inference on the given prompt.
    
    Args:
        prompt: Input text to generate from
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text response
    """
    pass
```

### 3. Testing

#### Run Existing Tests

```bash
# C++ inference test
./scripts/test_inference.sh

# Quick validation
./scripts/quick_test.sh

# Python tests (if pytest is installed)
pytest tests/
```

#### Add New Tests

For new features, add tests:
- C++ tests: Add to `test/` directory
- Python tests: Add to `tests/` directory with `test_` prefix

### 4. Documentation

Update documentation when:
- Adding new features
- Changing APIs
- Fixing bugs that need user awareness
- Modifying configuration options

Files to update:
- `README.md` - User-facing changes
- `docs/BUILD.md` - Build process changes
- `docs/FIXES_APPLIED.md` - Bug fixes and workarounds
- Code comments - Implementation details

### 5. Commit Messages

Use clear, descriptive commit messages:

```
<type>: <short summary> (max 50 chars)

<Detailed description if needed>
- Bullet points for multiple changes
- Reference issues with #issue_number

<Footer: Breaking changes, deprecations>
```

Types:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Code style (formatting, no logic change)
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Test additions/modifications
- `build:` - Build system changes
- `ci:` - CI configuration changes

Examples:
```
feat: add GPU multi-device support

- Allow specifying device_id for CUDA provider
- Add configuration option for device selection
- Update documentation

Closes #123
```

```
fix: resolve tensor lifecycle issue in decode loop

The token data vector was going out of scope before tensor creation,
causing garbage output. Fixed by maintaining persistent buffers.

Fixes #145
```

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows the style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex sections
- [ ] Documentation updated
- [ ] Tests added/updated and passing
- [ ] No unnecessary debug code or commented-out sections
- [ ] Build succeeds without warnings

### 2. Submit Pull Request

1. Push your branch to your fork
2. Open a Pull Request against `main` branch
3. Fill out the PR template completely
4. Link related issues

### 3. PR Title Format

```
<type>: <description>
```

Examples:
- `feat: add batch inference support`
- `fix: correct KV cache dimension handling`
- `docs: update GPU setup instructions`

### 4. PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Detailed list of changes
- Another change

## Testing
How was this tested?
- [ ] Manual testing
- [ ] Automated tests added
- [ ] Existing tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Build succeeds
- [ ] No breaking changes (or documented if necessary)

## Related Issues
Closes #123
Fixes #456
```

### 5. Review Process

- Maintainers will review your PR
- Address feedback and requested changes
- Push updates to the same branch
- Once approved, a maintainer will merge

## Areas for Contribution

### High Priority
- [ ] Batch inference support
- [ ] Additional model support (Qwen2, Llama)
- [ ] Performance optimizations
- [ ] Better error handling and messages
- [ ] Comprehensive test suite

### Documentation
- [ ] API documentation
- [ ] Tutorial notebooks
- [ ] Performance benchmarking guide
- [ ] Troubleshooting guide improvements

### Features
- [ ] Python CLI improvements
- [ ] Configuration validation
- [ ] Model download automation
- [ ] Quantization options (int8, int4)
- [ ] Multi-GPU inference

### Bug Fixes
Check the [issues page](https://github.com/sgowdaks/llm-inference/issues) for bugs to fix.

## Development Tips

### Building Efficiently

```bash
# Only rebuild changed files
cd build
make -j$(nproc)

# Clean build
cd build
make clean
cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
make -j$(nproc)
```

### Debugging C++

```bash
# Build with debug symbols
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
make -j$(nproc)

# Run with gdb
gdb --args ./bin/onnx_inference "test prompt"
```

### Python Development

```bash
# Install in development mode
pip install -e .

# Run with verbose output
python src/onnx_inference.py --config configs/config.json --prompt "test" -v
```

### Profiling

```bash
# C++ profiling
perf record -g ./build/bin/onnx_inference "test"
perf report

# Python profiling
python -m cProfile -o profile.stats src/onnx_inference.py --config configs/config.json --prompt "test"
python -m pstats profile.stats
```

## Questions?

- Open a discussion on GitHub
- Check existing issues and PRs
- Review documentation in `docs/`

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## Acknowledgments

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Git commit history

Thank you for contributing! 🎉
