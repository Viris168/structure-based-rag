"""Splits Markdown into sections at headings of one chosen level."""

from chunking.markdown_syntax import clean_heading, heading_pattern, is_fence_line
from chunking.models import Section


def split_on_heading(text: str, level: int) -> list[Section]:
    """Split Markdown at every heading of exactly `level`, ignoring code blocks.

    A `## Setup` line inside a ``` fence is code, not a heading, so it never splits.
    Text before the first heading becomes a Section with heading "".

    Args:
        text: Raw Markdown to split.
        level: Heading level to split on (2 for `##`, 3 for `###`).

    Returns:
        Sections in document order; whitespace-only slices are dropped.
    """
    pattern = heading_pattern(level)
    sections: list[Section] = []
    heading = ""
    lines: list[str] = []
    in_fence = False

    for line in text.splitlines():
        if is_fence_line(line):
            in_fence = not in_fence
        match = None if in_fence else pattern.match(line)
        if match:
            sections.append(Section(heading=heading, text="\n".join(lines).strip("\n")))
            heading = clean_heading(match.group(1))
            lines = [line]
        else:
            lines.append(line)
    sections.append(Section(heading=heading, text="\n".join(lines).strip("\n")))

    return [section for section in sections if section.text.strip()]
