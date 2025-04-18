# %%
import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

# %%
nbdirs = [
    os.path.join("docs/examples"),
]


def get_notebooks():
    nblist = []
    for nbdir in nbdirs:
        nblist += [
            os.path.join(nbdir, f) for f in os.listdir(nbdir) if f.endswith(".ipynb")
        ]
    return nblist


@pytest.mark.notebooks
@pytest.mark.parametrize("pth", get_notebooks())
def test_notebook_py(pth):
    with open(pth, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            assert (
                ep.preprocess(nb, {"metadata": {"path": "docs/examples"}}) is not None
            ), f"Got empty notebook for {os.path.basename(pth)}"
        except Exception as e:
            pytest.fail(reason=f"Failed executing {os.path.basename(pth)}: {e}")


if __name__ == "__main__":
    for notebook in get_notebooks():
        os.system(
            "jupyter nbconvert --clear-output --inplace "
            f"--ClearMetadataPreprocessor.enabled=True {notebook}"
        )
