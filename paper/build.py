"""Compile the paper without importing MLX or loading a model."""

from pathlib import Path

import typst


def main() -> None:
    directory = Path(__file__).resolve().parent
    output = directory / "output" / "pdf" / "morph32.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    typst.compile(str(directory / "main.typ"), output=str(output), root=str(directory))
    print(output)


if __name__ == "__main__":
    main()
