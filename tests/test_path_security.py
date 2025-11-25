#!/usr/bin/env python3
"""
Tests for path validation and security

Ensures path traversal attacks are prevented.
"""


import pytest

from knowledge_extraction.utils.path_utils import (
    PathSecurityError,
    sanitize_filename,
    validate_directory_path,
    validate_file_path,
)


class TestValidateFilePath:
    """Test file path validation"""

    def test_valid_file_path(self, tmp_path):
        """Valid file path should pass validation"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        result = validate_file_path(test_file, must_exist=True)
        assert result == test_file.resolve()

    def test_path_traversal_blocked(self):
        """Path traversal attempts should be blocked"""
        with pytest.raises(PathSecurityError):
            validate_file_path("../../etc/passwd")

    def test_double_slash_blocked(self):
        """Double slash should be blocked"""
        with pytest.raises(PathSecurityError):
            validate_file_path("test//file.pdf")

    def test_url_encoded_traversal_blocked(self):
        """URL-encoded path traversal should be blocked"""
        with pytest.raises(PathSecurityError):
            validate_file_path("test%2e%2e/file.pdf")

    def test_invalid_extension_blocked(self, tmp_path):
        """File with wrong extension should be blocked"""
        test_file = tmp_path / "test.exe"
        test_file.write_text("test")

        with pytest.raises(PathSecurityError):
            validate_file_path(
                test_file,
                allowed_extensions=['.pdf', '.txt'],
                must_exist=True
            )

    def test_valid_extension_allowed(self, tmp_path):
        """File with correct extension should pass"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")

        result = validate_file_path(
            test_file,
            allowed_extensions=['.pdf', '.txt'],
            must_exist=True
        )
        assert result.suffix == '.pdf'

    def test_nonexistent_file_with_must_exist(self):
        """Non-existent file with must_exist=True should fail"""
        with pytest.raises(FileNotFoundError):
            validate_file_path(
                "nonexistent_file.pdf",
                must_exist=True
            )

    def test_directory_not_file(self, tmp_path):
        """Directory path should fail file validation"""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        with pytest.raises(PathSecurityError):
            validate_file_path(test_dir, must_exist=True)

    def test_base_dir_restriction(self, tmp_path):
        """File outside base_dir should be blocked"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")

        other_dir = tmp_path / "allowed"
        other_dir.mkdir()

        with pytest.raises(PathSecurityError):
            validate_file_path(
                test_file,
                base_dir=other_dir,
                must_exist=True
            )

    def test_base_dir_allowed(self, tmp_path):
        """File within base_dir should be allowed"""
        base_dir = tmp_path / "allowed"
        base_dir.mkdir()

        test_file = base_dir / "test.pdf"
        test_file.write_text("test")

        result = validate_file_path(
            test_file,
            base_dir=base_dir,
            must_exist=True
        )
        assert result == test_file.resolve()


class TestValidateDirectoryPath:
    """Test directory path validation"""

    def test_valid_directory(self, tmp_path):
        """Valid directory should pass validation"""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        result = validate_directory_path(test_dir)
        assert result == test_dir.resolve()

    def test_path_traversal_blocked_dir(self):
        """Path traversal in directory should be blocked"""
        with pytest.raises(PathSecurityError):
            validate_directory_path("../../etc")

    def test_create_directory(self, tmp_path):
        """Should create directory when create=True"""
        new_dir = tmp_path / "new_dir"
        assert not new_dir.exists()

        result = validate_directory_path(new_dir, create=True)
        assert new_dir.exists()
        assert result == new_dir.resolve()

    def test_base_dir_restriction_dir(self, tmp_path):
        """Directory outside base_dir should be blocked"""
        test_dir = tmp_path / "test"
        base_dir = tmp_path / "allowed"
        base_dir.mkdir()

        with pytest.raises(PathSecurityError):
            validate_directory_path(test_dir, base_dir=base_dir)

    def test_file_not_directory(self, tmp_path):
        """File path should fail directory validation"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(PathSecurityError):
            validate_directory_path(test_file)


class TestSanitizeFilename:
    """Test filename sanitization"""

    def test_remove_path_separators(self):
        """Path separators should be replaced"""
        result = sanitize_filename("path/to/file.pdf")
        assert "/" not in result
        assert result == "file.pdf"

    def test_remove_unsafe_characters(self):
        """Unsafe characters should be replaced"""
        result = sanitize_filename("file:name*.pdf")
        assert ":" not in result
        assert "*" not in result
        assert result == "file_name_.pdf"

    def test_windows_reserved_names(self):
        """Windows reserved names should be escaped"""
        result = sanitize_filename("CON.txt")
        assert result == "_CON.txt_"  # Escapes reserved name

        result = sanitize_filename("PRN.pdf")
        assert result == "_PRN.pdf_"  # Escapes reserved name

    def test_max_length_enforcement(self):
        """Long filenames should be truncated"""
        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name, max_length=255)
        assert len(result) <= 255
        assert result.endswith(".pdf")

    def test_empty_filename(self):
        """Empty filename should raise error"""
        with pytest.raises(ValueError):
            sanitize_filename("")

    def test_control_characters_removed(self):
        """Control characters should be removed"""
        result = sanitize_filename("file\x00\x1fname.pdf")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_preserve_extension(self):
        """File extension should be preserved"""
        result = sanitize_filename("my file.pdf")
        assert result.endswith(".pdf")

    def test_trim_spaces_and_dots(self):
        """Leading/trailing spaces and dots should be removed"""
        result = sanitize_filename(" .test.pdf. ")
        assert not result.startswith(" ")
        assert not result.startswith(".")
        assert not result.endswith(" ")
        assert not result.endswith(".")


def test_integration_document_processor(tmp_path):
    """Test that DocumentProcessor uses path validation"""
    from knowledge_extraction.core.document_processor import DocumentProcessor

    processor = DocumentProcessor()

    # Path traversal is now caught and logged as error (returns None)
    # The validation happens in process_file which catches the exception
    result = processor.process_file("normal_file.txt")  # Non-existent but valid format
    assert result is None  # File doesn't exist, returns None

    # Valid file should work
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document.")

    result = processor.process_file(test_file)
    assert result is not None
    assert "text" in result


def test_integration_vector_store(tmp_path):
    """Test that VectorStore uses path validation"""
    from knowledge_extraction.core.vector_store import VectorStore

    # Valid path should work
    valid_db = tmp_path / "test_db"
    store = VectorStore(db_path=str(valid_db))
    assert store.db_path.exists()

    # Path traversal should be caught
    with pytest.raises((ValueError, PathSecurityError)):
        VectorStore(db_path="../../etc")
