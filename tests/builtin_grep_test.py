# -*- coding: utf-8 -*-
"""Grep tool test case."""
import os
import tempfile
from unittest.async_case import IsolatedAsyncioTestCase

from agentscope.message import ToolResultState
from agentscope.tool import Grep, ToolChunk
from agentscope.tool._builtin._backend import BackendBase, ExecResult
from agentscope.permission import (
    PermissionContext,
    PermissionBehavior,
    PermissionRule,
)


class GrepToolTest(IsolatedAsyncioTestCase):
    """The grep tool test case."""

    async def asyncSetUp(self) -> None:
        """The async setup method."""
        self.grep_tool = Grep()
        # Create a temporary directory with test files
        self.temp_dir = tempfile.mkdtemp()

        # Create test files
        with open(
            os.path.join(self.temp_dir, "test1.py"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("def hello():\n    print('Hello World')\n")

        with open(
            os.path.join(self.temp_dir, "test2.py"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("def goodbye():\n    print('Goodbye')\n")

        with open(
            os.path.join(self.temp_dir, "test.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("This is a text file\nHello from text\n")

        # Create subdirectory with files for glob pattern testing
        subdir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(subdir)
        with open(
            os.path.join(subdir, "nested.py"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("def nested():\n    print('Nested')\n")

    async def asyncTearDown(self) -> None:
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    async def test_tool_properties(self) -> None:
        """Test grep tool properties."""
        self.assertEqual(self.grep_tool.name, "Grep")
        self.assertIsInstance(self.grep_tool.description, str)
        self.assertIsInstance(self.grep_tool.input_schema, dict)
        self.assertFalse(self.grep_tool.is_mcp)
        self.assertTrue(self.grep_tool.is_read_only)
        self.assertTrue(self.grep_tool.is_concurrency_safe)

    async def test_check_permissions(self) -> None:
        """Test grep tool permission checking."""
        context = PermissionContext()
        tool_input = {"pattern": "hello"}
        decision = await self.grep_tool.check_permissions(tool_input, context)

        # Read/Glob/Grep are read-only, return PASSTHROUGH
        self.assertEqual(decision.behavior, PermissionBehavior.PASSTHROUGH)

    async def test_simple_search(self) -> None:
        """Test simple grep search."""
        chunk = await self.grep_tool(
            pattern="Hello",
            path=self.temp_dir,
            output_mode="files_with_matches",
        )

        self.assertEqual(chunk.state, ToolResultState.SUCCESS)

        content = chunk.content[0].text
        # Should find files containing "Hello"
        self.assertIn("test1.py", content)
        self.assertIn("test.txt", content)

    async def test_content_mode(self) -> None:
        """Test grep with content output mode."""
        chunk = await self.grep_tool(
            pattern="def",
            path=self.temp_dir,
            output_mode="content",
            type="py",
        )

        content = chunk.content[0].text

        # Should show matching lines
        self.assertIn("def hello", content)
        self.assertIn("def goodbye", content)

    async def test_case_insensitive(self) -> None:
        """Test case-insensitive search."""
        chunk = await self.grep_tool(
            pattern="HELLO",
            path=self.temp_dir,
            case_insensitive=True,
            output_mode="files_with_matches",
        )

        content = chunk.content[0].text
        self.assertIn("test1.py", content)

    async def test_no_matches(self) -> None:
        """Test search with no matches."""
        chunk = await self.grep_tool(
            pattern="NonExistentPattern",
            path=self.temp_dir,
        )

        self.assertIn("No matches found", chunk.content[0].text)

    async def test_type_filter(self) -> None:
        """Test filtering by file type."""
        chunk = await self.grep_tool(
            pattern="Hello",
            path=self.temp_dir,
            type="py",
            output_mode="files_with_matches",
        )

        content = chunk.content[0].text

        # Should only find .py files
        self.assertIn("test1.py", content)
        self.assertNotIn("test.txt", content)

    async def test_invalid_regex(self) -> None:
        """Test grep with invalid regex pattern."""
        chunk = await self.grep_tool(
            pattern="[invalid(regex",
            path=self.temp_dir,
        )

        self.assertEqual(chunk.state, "error")
        # ripgrep returns its own error message for regex parse errors
        self.assertIn("regex parse error", chunk.content[0].text)

    async def test_glob_pattern_with_subdirs(self) -> None:
        """Test glob pattern matching with subdirectories like **/*.py."""
        chunk = await self.grep_tool(
            pattern="def",
            path=self.temp_dir,
            glob="**/*.py",
            output_mode="files_with_matches",
        )

        content = chunk.content[0].text

        # Should find all .py files including in subdirectories
        self.assertIn("test1.py", content)
        self.assertIn("test2.py", content)
        self.assertIn("nested.py", content)
        # Should not find .txt files
        self.assertNotIn("test.txt", content)

    async def test_match_rule_path(self) -> None:
        """Test match_rule with search path patterns."""
        # Test matching explicit path
        self.assertTrue(
            await self.grep_tool.match_rule(
                self.temp_dir,
                {"path": self.temp_dir},
            ),
        )

        # Test wildcard pattern matching path
        parent_dir = os.path.dirname(self.temp_dir)
        self.assertTrue(
            await self.grep_tool.match_rule(
                parent_dir + "/**",
                {"path": self.temp_dir},
            ),
        )

        # Test non-matching path
        self.assertFalse(
            await self.grep_tool.match_rule(
                "/some/other/path/**",
                {"path": self.temp_dir},
            ),
        )

    async def test_match_rule_defaults_to_cwd(self) -> None:
        """Test match_rule defaults to cwd when no path is provided."""
        cwd = os.getcwd()

        # When no path provided, should match against cwd
        self.assertTrue(
            await self.grep_tool.match_rule(
                cwd,
                {"pattern": "hello"},
            ),
        )

        # Should not match a different path
        self.assertFalse(
            await self.grep_tool.match_rule(
                "/some/other/path",
                {"pattern": "hello"},
            ),
        )

    async def test_generate_suggestions_with_path(self) -> None:
        """Test generate_suggestions for grep with explicit path."""

        suggestions = await self.grep_tool.generate_suggestions(
            {"path": self.temp_dir, "pattern": "hello"},
        )

        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        self.assertIsInstance(suggestions[0], PermissionRule)

        # Should suggest directory pattern
        abs_path = os.path.abspath(self.temp_dir)
        expected_pattern = abs_path.rstrip("/") + "/**"
        suggestion_contents = [s.rule_content for s in suggestions]
        self.assertIn(expected_pattern, suggestion_contents)

    async def test_generate_suggestions_defaults_to_cwd(self) -> None:
        """Test generate_suggestions defaults to cwd when no path provided."""

        suggestions = await self.grep_tool.generate_suggestions(
            {"pattern": "hello"},
        )

        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

        cwd = os.getcwd()
        expected_pattern = os.path.abspath(cwd).rstrip("/") + "/**"
        suggestion_contents = [s.rule_content for s in suggestions]
        self.assertIn(expected_pattern, suggestion_contents)


class _UnstableRipgrepBackend(BackendBase):
    """A backend emulating ripgrep's ordering contract.

    Without ``--sort``, ripgrep's parallel walker returns matches in a
    different order on every invocation; this stub reproduces that by
    rotating the results once per call. With ``--sort``, ripgrep
    guarantees a stable order instead.
    """

    def __init__(self, search_dir: str) -> None:
        self.search_dir = search_dir
        self.calls = 0

    async def exec_shell(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        self.calls += 1
        search_path = command[-1]
        lines = []
        for name in sorted(os.listdir(self.search_dir)):
            file_path = os.path.join(self.search_dir, name)
            if not os.path.isfile(file_path):
                continue
            with open(file_path, encoding="utf-8") as f:
                first_line = f.readline().strip()
            lines.append(
                f"{os.path.join(search_path, name)}:1:{first_line}",
            )
        if lines and not any(
            arg == "--sort" or arg.startswith("--sort=") for arg in command
        ):
            shift = self.calls % len(lines)
            lines = lines[shift:] + lines[:shift]
        return ExecResult(
            0,
            ("\n".join(lines) + "\n").encode("utf-8"),
            b"",
        )

    async def read_file(self, path: str) -> bytes:
        raise NotImplementedError

    async def write_file(self, path: str, data: bytes) -> None:
        raise NotImplementedError


class GrepPaginationTest(IsolatedAsyncioTestCase):
    """Offset pagination must not lose or duplicate matches."""

    async def asyncSetUp(self) -> None:
        """The async setup method."""
        self.temp_dir = tempfile.mkdtemp()
        for i in range(6):
            with open(
                os.path.join(self.temp_dir, f"file{i}.py"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(f"NEEDLE{i}\n")

    async def asyncTearDown(self) -> None:
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @staticmethod
    def _needles(chunk: ToolChunk) -> list[str]:
        """Extract the NEEDLE marker of each matching line."""
        return [
            line.rsplit(":", 1)[-1]
            for line in chunk.content[0].text.splitlines()
            if ":NEEDLE" in line
        ]

    async def test_offset_pagination_keeps_every_match(self) -> None:
        """Paging with offset must return each match exactly once.

        rg returns matches in a different order on every invocation
        unless it is told to sort, so slicing with ``offset`` skips a
        different "first N" on every page and matches get lost from
        the paginated output while others appear on two pages.
        """
        grep_tool = Grep(backend=_UnstableRipgrepBackend(self.temp_dir))

        full = await grep_tool(
            pattern="NEEDLE",
            path=self.temp_dir,
            output_mode="content",
            head_limit=0,
        )
        page1 = await grep_tool(
            pattern="NEEDLE",
            path=self.temp_dir,
            output_mode="content",
            head_limit=3,
            offset=0,
        )
        page2 = await grep_tool(
            pattern="NEEDLE",
            path=self.temp_dir,
            output_mode="content",
            head_limit=3,
            offset=3,
        )

        # Every match must appear on exactly one page, in order.
        self.assertEqual(
            self._needles(page1) + self._needles(page2),
            self._needles(full),
        )

    async def test_results_are_sorted_by_path(self) -> None:
        """Results must come back in a stable, path-sorted order."""
        grep_tool = Grep()

        first = await grep_tool(
            pattern="NEEDLE",
            path=self.temp_dir,
            output_mode="content",
            head_limit=0,
        )
        second = await grep_tool(
            pattern="NEEDLE",
            path=self.temp_dir,
            output_mode="content",
            head_limit=0,
        )

        needles = self._needles(first)
        self.assertEqual(needles, sorted(needles))
        self.assertEqual(self._needles(second), needles)
