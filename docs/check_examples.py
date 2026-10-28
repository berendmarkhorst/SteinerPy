"""Execute the documentation notebooks; optionally refresh committed outputs."""

import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh source outputs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    examples = root / "docs/source/examples"
    paths = [root / "example.ipynb"] + [
        examples / f"{name}.ipynb"
        for name in ("connectivity", "costs_and_budgets", "constraints")
    ]
    output = root / "docs/_build/notebooks"
    output.mkdir(parents=True, exist_ok=True)
    for variable, directory in (
        ("IPYTHONDIR", "ipython"),
        ("JUPYTER_RUNTIME_DIR", "jupyter"),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        os.environ.setdefault(variable, str(output / directory))
    for path in paths:
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        for cell in book.cells:
            cell.metadata.pop("execution", None)
        client = NotebookClient(
            book,
            kernel_name="python3",
            timeout=60,
            allow_errors=False,
            record_timing=False,
            resources={"metadata": {"path": str(root)}},
        )
        client.km = client.create_kernel_manager()
        # A user-level kernelspec can otherwise select a different environment.
        client.km.kernel_spec.argv[0] = sys.executable
        client.execute()
        figures = 0
        for cell in book.cells:
            for result in cell.get("outputs", []):
                if result.output_type == "stream" and result.name == "stderr":
                    assert "Warning:" not in result.text, (
                        f"Warning in notebook output {path}:\n{result.text}"
                    )
                if any(mime.startswith("image/") for mime in result.get("data", {})):
                    settings = cell.get("metadata", {}).get("mystnb", {})
                    assert settings.get("image", {}).get("alt"), f"Missing alt text: {path}"
                    assert settings.get("figure", {}).get("caption"), (
                        f"Missing caption: {path}"
                    )
                    figures += 1
        assert figures, f"No figures produced: {path}"
        nbformat.validate(book)
        nbformat.write(book, output / path.name)
        if args.write:
            nbformat.write(book, path)
        print(f"OK {path.relative_to(root)} ({figures} figures)", flush=True)


if __name__ == "__main__":
    main()
