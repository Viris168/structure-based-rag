"""A small browser chat page for the assistant, built with Gradio."""

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from config.settings import DOC_ROUTE, UI_DESCRIPTION, UI_EXAMPLES, UI_HOST, UI_PORT, UI_TITLE
from generation.citation import format_citation
from pipeline.ask import run_question
from pipeline.state import GraphState
from ui.doc_links import doc_url
from ui.doc_page import render_doc_page


def _source_links(state: GraphState) -> list[str]:
    """Turn each cited source into a Markdown link to its section of the full document.

    Args:
        state: The finished graph state ("chunks" and "citations").

    Returns:
        One bullet per cited source, in retrieval order, e.g.
        "- [Source: MCP Documentation · Section: Installation](/doc/MCP_Documentation.md#installation)".
    """
    links: list[str] = []
    for chunk in state["chunks"]:
        label = format_citation(chunk.metadata)
        link = f"- [{label.strip('[]')}]({doc_url(chunk.metadata)})"
        if label in state["citations"] and link not in links:
            links.append(link)
    return links


def format_reply(question: str) -> str:
    """Answer one question and format it as Markdown for the chat window.

    Errors (missing API key, network problems, no credit) are shown in the
    chat instead of crashing the page, so the user can see what went wrong.

    Args:
        question: The user's message.

    Returns:
        The answer, then a bulleted "Sources" list of links if the answer cited any.
    """
    try:
        state = run_question(question)
    except Exception as error:  # shown to the user rather than stopping the server
        return f"**Error:** {error}"
    reply = state["answer"]
    links = _source_links(state)
    if links:
        reply += "\n\n**Sources**\n" + "\n".join(links)
    return reply


def _respond(message: str, history: list[dict[str, str]]) -> str:
    """Gradio callback: each question is answered on its own.

    The assistant has no memory of earlier turns; every question is searched
    and answered fresh, so `history` is accepted (Gradio passes it) but unused.

    Args:
        message: The user's new message.
        history: Earlier messages in the chat.

    Returns:
        The formatted reply.
    """
    return format_reply(message)


def build_chat_app() -> gr.ChatInterface:
    """Create the chat page: a chat box, example questions, title, and description.

    Returns:
        The Gradio app, ready to mount.
    """
    return gr.ChatInterface(
        fn=_respond,
        title=UI_TITLE,
        description=UI_DESCRIPTION,
        examples=UI_EXAMPLES,
        analytics_enabled=False,
    )


def _show_document(source_path: str) -> HTMLResponse:
    """Serve the full-document page that source links point to.

    Args:
        source_path: Filename of a document in the docs folder.

    Returns:
        The rendered document.

    Raises:
        HTTPException: 404 if the file is not one of the docs.
    """
    page = render_doc_page(source_path)
    if page is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return HTMLResponse(page)


def build_server() -> FastAPI:
    """Combine the chat page (at "/") and the document pages (at DOC_ROUTE) in one web app.

    The document route is added before the chat is mounted, so it takes
    priority over Gradio's catch-all "/" route.

    Returns:
        The web app, ready to serve.
    """
    server = FastAPI()
    server.add_api_route(f"{DOC_ROUTE}/{{source_path}}", _show_document, response_class=HTMLResponse)
    return gr.mount_gradio_app(server, build_chat_app(), path="/")


def launch_chat_app() -> None:
    """Serve the chat at http://UI_HOST:UI_PORT until Ctrl+C."""
    print(f"Chat running at http://{UI_HOST}:{UI_PORT}  (Ctrl+C to stop)")
    uvicorn.run(build_server(), host=UI_HOST, port=UI_PORT, log_level="warning")
