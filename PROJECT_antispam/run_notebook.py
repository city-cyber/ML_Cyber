"""Выполнить все ячейки и сохранить ноутбук с таблицами, графиками и выводами."""

import os
import sys
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parent
    target = root / "project/PROJECT.ipynb"
    notebook = nbformat.read(target, as_version=4)
    nbformat.validate(notebook)
    # Не требует установки kernelspec за пределами проекта.
    manager = KernelSpecManager()
    spec = manager.get_kernel_spec("python3")
    spec.argv[0] = sys.executable
    class LocalKernelSpecManager(KernelSpecManager):
        def get_kernel_spec(self, kernel_name):
            return spec
    client = NotebookClient(
        notebook, timeout=1800, kernel_name="python3", allow_errors=False,
        resources={"metadata": {"path": str(root / "project")}},
    )
    client.kernel_manager_class = "jupyter_client.KernelManager"
    # Папки runtime/config остаются в рабочем проекте.
    os.environ["JUPYTER_RUNTIME_DIR"] = str(root / ".jupyter/runtime")
    os.environ["IPYTHONDIR"] = str(root / ".jupyter/ipython")
    root.joinpath(".jupyter/runtime").mkdir(parents=True, exist_ok=True)
    client.create_kernel_manager()
    client.km.kernel_spec_manager = LocalKernelSpecManager()
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
    client.on_cell_start = progress
    try:
        client.execute()
    finally:
        nbformat.write(notebook, target)
    nbformat.validate(notebook)
    errors = [out for cell in notebook.cells if cell.cell_type == "code" for out in cell.outputs if out.output_type == "error"]
    if errors:
        raise RuntimeError(f"В ноутбуке {len(errors)} ошибок")
    print(f"Completed: {target}")


if __name__ == "__main__":
    main()
