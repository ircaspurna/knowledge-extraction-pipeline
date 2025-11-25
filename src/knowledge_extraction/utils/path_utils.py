#!/usr/bin/env python3
"""
Path validation and sanitization utilities

Prevents path traversal attacks and ensures safe file operations.
"""

import os
import re
from pathlib import Path


class PathSecurityError(Exception):
    """Raised when a path fails security validation"""
    pass


def validate_file_path(
    file_path: str | Path,
    allowed_extensions: list[str] | None = None,
    must_exist: bool = False,
    base_dir: Path | None = None
) -> Path:
    """
    Validate and sanitize a file path for safe operations.

    Security checks:
    - Prevents path traversal (../, ../../, etc.)
    - Validates file extension if specified
    - Optionally ensures path is within base directory
    - Resolves symlinks to prevent directory escape

    Args:
        file_path: Path to validate
        allowed_extensions: List of allowed extensions (e.g., ['.pdf', '.txt'])
                          If None, all extensions allowed
        must_exist: If True, raises error if file doesn't exist
        base_dir: If provided, ensures path is within this directory

    Returns:
        Validated and resolved Path object

    Raises:
        PathSecurityError: If path fails security validation
        FileNotFoundError: If must_exist=True and file doesn't exist
        ValueError: If inputs are invalid

    Examples:
        >>> validate_file_path("paper.pdf", allowed_extensions=['.pdf'])
        Path('paper.pdf')

        >>> validate_file_path("../../etc/passwd")  # Raises PathSecurityError
    """
    # Validate input
    if not file_path:
        raise ValueError("file_path cannot be None or empty")

    # Convert to Path
    if isinstance(file_path, str):
        path = Path(file_path)
    elif isinstance(file_path, Path):
        path = file_path
    else:
        raise ValueError(f"file_path must be str or Path, got {type(file_path).__name__}")

    # Resolve to absolute path (expands ~, .., symlinks)
    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise PathSecurityError(f"Failed to resolve path: {e}")

    # Check for path traversal attempts in the original path string
    path_str = str(file_path)
    suspicious_patterns = [
        '../',           # Parent directory traversal
        '..\\',          # Windows parent traversal
        '//',            # Double slash (can bypass filters)
        '\\\\',          # Double backslash
        '%2e%2e',        # URL-encoded ..
        '..%2f',         # Mixed encoding
        '%252e',         # Double-encoded
    ]

    path_lower = path_str.lower()
    for pattern in suspicious_patterns:
        if pattern in path_lower:
            raise PathSecurityError(
                f"Path contains suspicious pattern '{pattern}': {file_path}"
            )

    # Check if path is within base directory
    if base_dir is not None:
        base_dir_resolved = Path(base_dir).resolve()

        try:
            # Check if resolved path is relative to base_dir
            resolved_path.relative_to(base_dir_resolved)
        except ValueError:
            raise PathSecurityError(
                f"Path {file_path} is outside allowed directory {base_dir}"
            )

    # Validate file extension
    if allowed_extensions is not None:
        if not isinstance(allowed_extensions, list):
            raise ValueError("allowed_extensions must be a list")

        # Normalize extensions (add . if missing)
        allowed_exts = [
            ext if ext.startswith('.') else f'.{ext}'
            for ext in allowed_extensions
        ]

        if resolved_path.suffix.lower() not in [e.lower() for e in allowed_exts]:
            raise PathSecurityError(
                f"File extension '{resolved_path.suffix}' not allowed. "
                f"Allowed: {allowed_exts}"
            )

    # Check existence if required
    if must_exist and not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {resolved_path}")

    # Ensure it's a file (not a directory) if it exists
    if resolved_path.exists() and not resolved_path.is_file():
        raise PathSecurityError(f"Path exists but is not a file: {resolved_path}")

    return resolved_path


def validate_directory_path(
    dir_path: str | Path,
    create: bool = False,
    base_dir: Path | None = None
) -> Path:
    """
    Validate and sanitize a directory path for safe operations.

    Security checks:
    - Prevents path traversal
    - Optionally ensures path is within base directory
    - Resolves symlinks

    Args:
        dir_path: Directory path to validate
        create: If True, creates directory if it doesn't exist
        base_dir: If provided, ensures path is within this directory

    Returns:
        Validated and resolved Path object

    Raises:
        PathSecurityError: If path fails security validation
        ValueError: If inputs are invalid

    Examples:
        >>> validate_directory_path("./output", create=True)
        Path('/full/path/to/output')
    """
    # Validate input
    if not dir_path:
        raise ValueError("dir_path cannot be None or empty")

    # Convert to Path
    if isinstance(dir_path, str):
        path = Path(dir_path)
    elif isinstance(dir_path, Path):
        path = dir_path
    else:
        raise ValueError(f"dir_path must be str or Path, got {type(dir_path).__name__}")

    # Resolve to absolute path
    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise PathSecurityError(f"Failed to resolve path: {e}")

    # Check for path traversal (same as file validation)
    path_str = str(dir_path)
    suspicious_patterns = ['../', '..\\', '//', '\\\\', '%2e%2e', '..%2f', '%252e']

    path_lower = path_str.lower()
    for pattern in suspicious_patterns:
        if pattern in path_lower:
            raise PathSecurityError(
                f"Path contains suspicious pattern '{pattern}': {dir_path}"
            )

    # Check if path is within base directory
    if base_dir is not None:
        base_dir_resolved = Path(base_dir).resolve()

        try:
            resolved_path.relative_to(base_dir_resolved)
        except ValueError:
            raise PathSecurityError(
                f"Path {dir_path} is outside allowed directory {base_dir}"
            )

    # Create directory if requested
    if create:
        try:
            resolved_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise PathSecurityError(f"Failed to create directory: {e}")

    # Ensure it's a directory if it exists
    if resolved_path.exists() and not resolved_path.is_dir():
        raise PathSecurityError(f"Path exists but is not a directory: {resolved_path}")

    return resolved_path


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename by removing/replacing unsafe characters.

    Removes:
    - Path separators (/, \\)
    - Null bytes
    - Control characters
    - Reserved names (Windows: CON, PRN, AUX, NUL, COM1-9, LPT1-9)

    Args:
        filename: Filename to sanitize
        max_length: Maximum filename length (default: 255, typical filesystem limit)

    Returns:
        Sanitized filename

    Examples:
        >>> sanitize_filename("my/file.pdf")
        'my_file.pdf'

        >>> sanitize_filename("CON.txt")  # Windows reserved name
        '_CON_.txt'
    """
    if not filename:
        raise ValueError("filename cannot be None or empty")

    # Remove path components (take only the basename)
    filename = os.path.basename(filename)

    # Remove null bytes
    filename = filename.replace('\0', '')

    # Remove control characters (0x00-0x1F, 0x7F-0x9F)
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)

    # Replace path separators and other unsafe characters
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in unsafe_chars:
        filename = filename.replace(char, '_')

    # Check for Windows reserved names
    reserved_names = [
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    ]

    name_without_ext = os.path.splitext(filename)[0].upper()
    if name_without_ext in reserved_names:
        filename = f'_{filename}_'

    # Trim to max length (preserve extension if possible)
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        name = name[:max_length - len(ext) - 1]
        filename = name + ext

    # Ensure filename doesn't start/end with spaces or dots
    filename = filename.strip(' .')

    # Ensure we still have a valid filename
    if not filename:
        filename = 'unnamed_file'

    return filename
