"""Breaks text that is too long for the embedding model into smaller whole pieces."""

from chunking.markdown_syntax import is_fence_line
from chunking.text_size import count_tokens


def _split_blocks(text: str) -> list[str]:
    """Split text at blank lines, keeping each fenced code block in one piece.

    Tables have no blank lines inside them, so they also stay whole.

    Args:
        text: Raw Markdown.

    Returns:
        Paragraphs, lists, tables, and code blocks in order.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if is_fence_line(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack(units: list[str], max_tokens: int, joiner: str) -> list[str]:
    """Greedily group consecutive units so each group fits within max_tokens.

    A single unit larger than max_tokens becomes a group of its own.

    Args:
        units: Blocks or lines, in order.
        max_tokens: Token ceiling for each group.
        joiner: Separator placed between units ("\\n\\n" for blocks, "\\n" for lines).

    Returns:
        The joined groups in order.
    """
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            groups.append(joiner.join(current))
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        groups.append(joiner.join(current))
    return groups


def _split_by_lines(block: str, max_tokens: int) -> list[str]:
    """Split one oversized block (a long list or code block) at line boundaries.

    A fenced code block is split inside its fences and each piece is re-wrapped
    with the original opening and closing fence, so every piece stays valid.

    Args:
        block: A single block that exceeds max_tokens.
        max_tokens: Token ceiling for each piece.

    Returns:
        Pieces in order.
    """
    lines = block.splitlines()
    is_fenced = len(lines) >= 2 and is_fence_line(lines[0]) and is_fence_line(lines[-1])
    if not is_fenced:
        return _pack(lines, max_tokens, "\n")
    opening, closing = lines[0], lines[-1]
    body_budget = max_tokens - count_tokens(f"{opening}\n{closing}")
    return [f"{opening}\n{body}\n{closing}" for body in _pack(lines[1:-1], body_budget, "\n")]


def pack_paragraphs(text: str, max_tokens: int) -> list[str]:
    """Split text into parts that each fit within max_tokens.

    Splits at blank lines first; any single block still too large is split
    at line boundaries. Only a single line longer than the limit can exceed it.

    Args:
        text: Raw Markdown that exceeds the limit.
        max_tokens: Token ceiling for each part.

    Returns:
        Parts in document order.
    """
    units: list[str] = []
    for block in _split_blocks(text):
        if count_tokens(block) > max_tokens:
            units.extend(_split_by_lines(block, max_tokens))
        else:
            units.append(block)
    return _pack(units, max_tokens, "\n\n")
