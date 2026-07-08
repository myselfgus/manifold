"""Entry-point `vdlp-manifold` — delega para o CLI do pipeline."""
import os
import sys


def main() -> int:
    # run_pipeline.py vive em python/ (um nível acima do pacote)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, ".."))
    from run_pipeline import main as _main  # type: ignore

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
