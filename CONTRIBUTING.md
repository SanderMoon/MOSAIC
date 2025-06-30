# Contributing to MOSAIC

Thank you for your interest in contributing to MOSAIC! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [sander.moonemans@gmail.com](mailto:sander.moonemans@gmail.com).

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/MOSAIC.git
   cd MOSAIC
   ```

3. Add the upstream repository:

   ```bash
   git remote add upstream https://github.com/SanderMoon/MOSAIC.git
   ```

## Development Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode:

   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

4. Install additional dependencies:

   ```bash
   pip install git+https://github.com/salaniz/pycocoevalcap.git
   ```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix identified issues in the codebase
- **Feature additions**: Add new functionality to the framework
- **Documentation improvements**: Enhance documentation, examples, or tutorials
- **Performance optimizations**: Improve code efficiency or speed
- **Test coverage**: Add or improve tests

### Before You Start

1. Check the [issue tracker](https://github.com/SanderMoon/MOSAIC/issues) to see if your idea or bug report already exists
2. For major changes, please open an issue first to discuss your proposed changes
3. Make sure you have the latest version of the main branch

## Pull Request Process

1. **Create a branch**: Create a new branch from `main` for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**: Implement your changes following the code style guidelines

3. **Test your changes**: Ensure all tests pass and add new tests if necessary:

   ```bash
   pytest
   ```

4. **Run code quality checks**:

   ```bash
   black .
   isort .
   flake8 .
   mypy src/
   ```

5. **Commit your changes** with a clear commit message:

   ```bash
   git commit -m "Add feature: brief description of changes"
   ```

6. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**: Open a pull request against the `main` branch

### Pull Request Guidelines

- **Clear description**: Provide a clear description of what your PR does
- **Reference issues**: Link to any relevant issues using "Fixes #issue_number"
- **Update documentation**: Update relevant documentation if needed
- **One feature per PR**: Keep pull requests focused on a single feature or fix

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions focused and relatively small
- Use meaningful variable and function names

### Example Function

```python
def process_image_features(
    features: torch.Tensor, 
    max_patches: int = 1000
) -> torch.Tensor:
    """
    Process image features by selecting top patches.
    
    Args:
        features: Input tensor of shape [batch, patches, dim]
        max_patches: Maximum number of patches to keep
        
    Returns:
        Processed features tensor
    """
    if features.size(1) > max_patches:
        features = features[:, :max_patches, :]
    return features
```

## Reporting Issues

When reporting issues, please include:

1. **Clear title**: Brief description of the issue
2. **Environment details**: Python version, OS, relevant package versions
3. **Steps to reproduce**: Detailed steps to reproduce the issue
4. **Expected behavior**: What you expected to happen
5. **Actual behavior**: What actually happened
6. **Code example**: Minimal reproducible example if applicable
7. **Error messages**: Full error messages and stack traces

### Issue Templates

Use the provided issue templates when available:

- Bug report
- Feature request
- Documentation improvement

## Development Workflow

1. **Stay up to date**: Regularly sync with upstream:

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create feature branches** from the latest main branch

3. **Make small, focused commits** with clear commit messages

4. **Test thoroughly** before submitting PRs

## Recognition

Contributors will be acknowledged in the project. Significant contributions may be recognized in the project documentation or release notes.

## Getting Help

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: Search existing issues or create a new one
- **Email**: Contact [sander.moonemans@gmail.com](mailto:sander.moonemans@gmail.com) for questions

## License

By contributing to MOSAIC, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to MOSAIC! Your contributions help advance computational pathology research.

