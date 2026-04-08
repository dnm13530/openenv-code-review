"""Server entry point for multi-mode deployment.

Re-exports the FastAPI app from src.main and provides a CLI entry point.
"""

from src.main import app  # noqa: F401 — re-export for uvicorn discovery


def main() -> None:
    """CLI entry point: start the uvicorn server."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
