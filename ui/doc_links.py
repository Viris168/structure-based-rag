"""Builds links from a source citation to its section in the full-document page."""

import re
from urllib.parse import quote

from chunking.metadata import Metadata
from config.settings import ANCHOR_SEPARATOR, DOC_ROUTE, SUBSECTION_JOINER

_NON_WORD_PATTERN = re.compile(r"[^a-z0-9]+")


def heading_anchor(*headings: str) -> str:
    """Turn a heading path into the ID used on the document page.

    Including the parent section keeps IDs unique when two sections share a
    subsection name (e.g. "Linux" under both "Virtual Machines" and "Tests").

    Args:
        headings: The section heading, then optionally the subsection heading.

    Returns:
        e.g. "server-setup--authentication" for ("Server Setup", "Authentication").
    """
    slugs = [_NON_WORD_PATTERN.sub("-", heading.lower()).strip("-") for heading in headings if heading]
    return ANCHOR_SEPARATOR.join(slugs)


def doc_url(metadata: Metadata) -> str:
    """Link to the chunk's section on the full-document page.

    A merged chunk ("Colors / Restoring PATH") links to its first subsection.

    Args:
        metadata: The chunk's stored metadata.

    Returns:
        e.g. "/doc/MCP_Documentation.md#server-setup--authentication".
    """
    subsection = str(metadata["subsection"]).split(SUBSECTION_JOINER)[0]
    anchor = heading_anchor(str(metadata["section_title"]), subsection)
    return f"{DOC_ROUTE}/{quote(str(metadata['source_path']))}#{anchor}"
