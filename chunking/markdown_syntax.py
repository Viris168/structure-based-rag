"""Recognises the Markdown line types the splitters care about: fences and headings."""

import re

_FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
_ANY_HEADING_PATTERN = re.compile(r"^#{1,6}\s")
# ![alt](url) and ![alt][ref] — badges and images carry no meaning in a heading.
_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\](\([^)]*\)|\[[^\]]*\])")
# [text](url) and [text][ref] — keep only the visible text.
_LINK_PATTERN = re.compile(r"\[([^\]]*)\](\([^)]*\)|\[[^\]]*\])")


def is_fence_line(line: str) -> bool:
    """Return True if the line opens or closes a fenced code block.

    Args:
        line: A single line of Markdown.

    Returns:
        True for lines starting with ``` or ~~~ (after optional indentation).
    """
    return bool(_FENCE_PATTERN.match(line))


def heading_pattern(level: int) -> re.Pattern[str]:
    """Build a regex that matches headings of exactly one level.

    `##` must not match `###`, so the marks are followed by required whitespace.

    Args:
        level: Number of `#` marks, e.g. 2 for `## Heading`.

    Returns:
        A compiled pattern whose group 1 is the heading text.
    """
    return re.compile(rf"^#{{{level}}}\s+(.+?)\s*$")


def clean_heading(text: str) -> str:
    """Reduce a heading to readable words: drop images, unwrap links.

    `Node Version Manager [![Build](badge.svg)][3]` becomes `Node Version Manager`,
    which keeps heading paths short and meaningful when embedded.

    Args:
        text: Heading text as written in the Markdown.

    Returns:
        The heading's plain text with whitespace collapsed; may be "".
    """
    text = _IMAGE_PATTERN.sub("", text)
    text = _LINK_PATTERN.sub(r"\1", text)
    return " ".join(text.split())


def has_content(text: str) -> bool:
    """Return True if the text contains anything besides headings and blank lines.

    Used to drop slices like a lone `# Title` that would make empty chunks.

    Args:
        text: A slice of Markdown.

    Returns:
        True if at least one non-blank, non-heading line exists.
    """
    return any(line.strip() and not _ANY_HEADING_PATTERN.match(line) for line in text.splitlines())
