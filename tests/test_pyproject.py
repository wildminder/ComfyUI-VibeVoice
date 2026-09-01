"""Tests for pyproject.toml - V3 compliance."""

import os
import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _load_pyproject():
    """Load pyproject.toml."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject_path = os.path.join(root, "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)


def _requirements_entries():
    """requirements.txt dependency lines (non-comment, non-empty)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "requirements.txt"), "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def _dep_name(spec):
    """Package name from a dependency specifier ("transformers>=4.51.3" -> "transformers")."""
    return spec.split(">=")[0].split("==")[0].split("[")[0].strip().lower()


class TestPyprojectCompliance:
    """Test pyproject.toml V3 compliance."""

    def test_requires_python(self):
        data = _load_pyproject()
        assert "requires-python" in data["project"]
        assert ">=3.10" in data["project"]["requires-python"]

    def test_requires_comfyui(self):
        data = _load_pyproject()
        assert "tool" in data
        assert "comfy" in data["tool"]
        assert "requires-comfyui" in data["tool"]["comfy"]
        assert ">=0.28.0" in data["tool"]["comfy"]["requires-comfyui"]

    def test_publisher_id(self):
        data = _load_pyproject()
        assert "PublisherId" in data["tool"]["comfy"]
        assert data["tool"]["comfy"]["PublisherId"]

    def test_display_name(self):
        data = _load_pyproject()
        assert "DisplayName" in data["tool"]["comfy"]

    def test_dependencies_present(self):
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].strip() for d in deps]
        assert "torch" in dep_names
        assert "transformers" in dep_names
        assert "huggingface_hub" in dep_names

    def test_project_urls(self):
        data = _load_pyproject()
        assert "urls" in data["project"]
        urls = data["project"]["urls"]
        assert "Repository" in urls

    def test_classifiers(self):
        data = _load_pyproject()
        assert "classifiers" in data["project"]
        assert len(data["project"]["classifiers"]) > 0

    def test_version(self):
        data = _load_pyproject()
        assert "version" in data["project"]


class TestAudioDependencySurface:
    """Phase 4: librosa/scipy must be optional; torchaudio is the primary audio lib."""

    @staticmethod
    def _dep_names(data):
        deps = data["project"]["dependencies"]
        return [d.split(">=")[0].split("==")[0].split("[")[0].strip() for d in deps]

    def test_librosa_not_in_hard_dependencies(self):
        data = _load_pyproject()
        assert "librosa" not in self._dep_names(data), \
            "librosa must be optional, not a hard dependency"

    def test_scipy_not_in_hard_dependencies(self):
        data = _load_pyproject()
        assert "scipy" not in self._dep_names(data), \
            "scipy must be optional (torchaudio is the primary resampler)"

    def test_torchaudio_in_hard_dependencies(self):
        data = _load_pyproject()
        assert "torchaudio" in self._dep_names(data), \
            "torchaudio is the primary audio library and must stay a hard dependency"

    def test_soundfile_in_hard_dependencies(self):
        data = _load_pyproject()
        assert "soundfile" in self._dep_names(data), \
            "soundfile is the primary audio encoder and must stay a hard dependency"

    def test_audio_extra_optional_group_declared(self):
        data = _load_pyproject()
        extras = data["project"].get("optional-dependencies", {})
        assert "audio-extra" in extras, "optional-dependencies.audio-extra must exist"
        extra_names = [d.split(">=")[0].split("==")[0].strip() for d in extras["audio-extra"]]
        assert "scipy" in extra_names
        assert "librosa" in extra_names

    def test_requirements_txt_has_no_librosa(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "requirements.txt"), "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        names = [ln.split(">=")[0].split("==")[0].strip() for ln in lines]
        assert "librosa" not in names
        assert "scipy" not in names
        assert "torchaudio" in names


class TestRequirementsPyprojectParity:
    """requirements.txt must be a dependency-name subset of pyproject.toml.

    requirements.txt serves git-clone / ComfyUI-Manager installs (only the
    packages NOT bundled with ComfyUI core); pyproject.toml ``dependencies``
    is the full declared surface used by the Comfy registry. A name present
    in requirements.txt but missing from pyproject would silently break
    registry installs — this guards against that drift.
    """

    def test_requirements_are_pyproject_subset(self):
        data = _load_pyproject()
        pyproject_names = {_dep_name(d) for d in data["project"]["dependencies"]}
        for entry in _requirements_entries():
            name = _dep_name(entry)
            assert name in pyproject_names, (
                f"requirements.txt lists '{name}' but pyproject.toml "
                f"dependencies does not — registry installs would miss it. "
                f"Add it to [project] dependencies (or remove it from "
                f"requirements.txt if ComfyUI bundles it)."
            )

    def test_version_pins_agree(self):
        """Where both files pin the same package, the pins must not conflict."""
        data = _load_pyproject()
        pyproject = {_dep_name(d): d for d in data["project"]["dependencies"]}
        for entry in _requirements_entries():
            name = _dep_name(entry)
            if name in pyproject and ">=" in entry and ">=" in pyproject[name]:
                req_floor = entry.split(">=")[1].split(",")[0].strip()
                py_floor = pyproject[name].split(">=")[1].split(",")[0].strip()
                assert req_floor == py_floor, (
                    f"Version floors disagree for '{name}': "
                    f"requirements.txt wants >= {req_floor}, "
                    f"pyproject.toml wants >= {py_floor}."
                )
