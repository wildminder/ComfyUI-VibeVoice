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


def _version_tuple(version: str):
    """(major, minor, patch) as ints, tolerating a local tag like "4.57.6+cu126"."""
    parts = version.split("+")[0].split(".")
    numbers = []
    for part in parts[:3]:
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers)


def _installed_transformers_version():
    """The transformers version the suite is actually running against, or None."""
    try:
        import transformers
    except ImportError:
        return None
    return transformers.__version__


def _version_in_range(version: str, spec: str) -> bool:
    """Does ``version`` satisfy a ``>=floor,<cap`` style specifier?"""
    current = _version_tuple(version)
    for part in spec.split(",", 1)[1:]:
        part = part.strip()
        if part.startswith(">="):
            if current < _version_tuple(part[2:].strip()):
                return False
        elif part.startswith(">"):
            if current <= _version_tuple(part[1:].strip()):
                return False
        elif part.startswith("<="):
            if current > _version_tuple(part[2:].strip()):
                return False
        elif part.startswith("<"):
            if current >= _version_tuple(part[1:].strip()):
                return False
        elif part.startswith("=="):
            if current[:len(_version_tuple(part[2:].strip()))] != _version_tuple(part[2:].strip()):
                return False
    return True


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


class TestTransformersRangeHonesty:
    """S5.1: the declared ``transformers`` range must be internally consistent.

    ``transformers`` is the one dependency whose internal API this node adapts to
    (the realtime path carries a legacy pickled ``DynamicCache`` across a 4.x ->
    5.x cache refactor). The declaration therefore has to be *agreed* rather
    than merely present.

    The current declaration is unbounded (``"transformers"`` in both files), a
    deliberate decision: a registry install and a Manager/git-clone install must
    resolve the same major, and the two files used to disagree — pyproject.toml
    without a cap, requirements.txt with ``>=5.3.0,<5.4`` — so the two install
    paths landed on different transformers. Whether a bound is present is
    therefore a *choice*; whether the two files express the *same* choice is a
    contract, and that is what these tests assert.

    What a bound buys is still checked: when one IS declared, every major it
    admits must be in :attr:`MEASURED_MAJORS`, and the version the suite
    actually runs against must satisfy the declaration. When none is declared,
    the installed major must itself be measured — the only remaining guard
    against the silent-cache-shim failure described above.

    The measured set is recorded here as a literal rather than read from a
    development document: it is a fact about the shipped code, and the published
    suite must not depend on files a clone does not have. The two-generation
    record itself, the exact commands and the reproduction steps live in the
    support-matrix section of the README.
    """

    # The ``transformers`` major lines this node has actually been exercised on.
    # Widen MEASURED_MAJORS only after running the real-checkpoint generation
    # tests against the new major.
    MEASURED_MAJORS = ("4", "5")

    @classmethod
    def _spec(cls) -> str:
        data = _load_pyproject()
        specs = [d for d in data["project"]["dependencies"] if _dep_name(d) == "transformers"]
        assert len(specs) == 1, (
            f"pyproject.toml must declare transformers exactly once, found {specs}"
        )
        return specs[0]

    @classmethod
    def _requirement_spec(cls) -> str:
        entries = [e for e in _requirements_entries() if _dep_name(e) == "transformers"]
        assert len(entries) == 1, (
            f"requirements.txt must list transformers exactly once, found {entries}"
        )
        return entries[0]

    def test_transformers_spec_agrees_across_files(self):
        """pyproject.toml and requirements.txt must express the SAME choice.

        This is the invariant that survives a decision to bound or not to
        bound. Comparing the whole spec (not just the cap) also catches a
        floor added on one side only.
        """
        py_spec = self._spec()
        req_spec = self._requirement_spec()
        assert py_spec == req_spec, (
            f"The transformers declaration disagrees across the two install "
            f"paths: pyproject.toml declares {py_spec!r} while requirements.txt "
            f"declares {req_spec!r}. A Comfy registry install resolves the "
            f"former and a Manager/git-clone install resolves the latter, so "
            f"the same workflow runs on two different transformers majors. "
            f"Make the two files identical."
        )

    def test_requirements_txt_bound_matches_pyproject(self):
        """The cap, when one is declared, must be the same cap in both files.

        Narrower than :meth:`test_transformers_spec_agrees_across_files` on
        purpose: this fails with a message that names the cap specifically,
        which is the field that actually broke (pyproject had dropped it while
        requirements.txt kept it).
        """
        py_spec = self._spec()
        py_cap = py_spec.split("<")[-1].strip() if "<" in py_spec else None
        req_spec = self._requirement_spec()
        req_cap = req_spec.split("<")[-1].strip() if "<" in req_spec else None
        assert req_cap == py_cap, (
            f"transformers upper bounds disagree: pyproject.toml has {py_cap!r}, "
            f"requirements.txt has {req_cap!r}. A Manager/git-clone install "
            f"would resolve a different major than a registry install."
        )

    def test_range_matches_the_version_the_suite_runs_against(self):
        """The installed transformers must satisfy the declared range.

        This is the drift guard. It is allowed to be *outside* the range only when
        that major is recorded here as measured but not admitted — a development
        machine is allowed to run ahead of what the package declares, an
        installable version is not.
        """
        spec = self._spec()
        installed = _installed_transformers_version()
        assert installed is not None, (
            "transformers is not importable here, so the declared range cannot be "
            "checked against the version the suite actually runs against."
        )

        if _version_in_range(installed, spec):
            return

        major = installed.split(".")[0]
        assert major in self.MEASURED_MAJORS, (
            f"The suite is running transformers {installed}, which the declared "
            f"range '{spec}' does not admit, and {major}.x is not recorded in "
            f"MEASURED_MAJORS. Either the range is wrong or the measured set is "
            f"stale — an unrecorded major is exactly how F12 happened."
        )

    def test_every_admitted_major_has_been_measured(self):
        """Whatever the declaration admits must be recorded as measured.

        With a cap this is the historical check: each major below the cap must
        appear in :attr:`MEASURED_MAJORS`. Without a cap the declaration admits
        every future major, so that loop is vacuous and the meaningful guard is
        that the major the suite actually runs against was measured — an
        unbounded declaration must not become a way to run unmeasured code.
        """
        spec = self._spec()
        majors = set()
        for part in spec.split(",", 1)[1:]:
            part = part.strip()
            if part.startswith("<"):
                majors.add(part[1:].strip().split(".")[0])

        if majors:
            for major in sorted(majors):
                assert major in self.MEASURED_MAJORS, (
                    f"The declared range admits transformers {major}.x, but "
                    f"{major}.x is not in MEASURED_MAJORS. Widening the range "
                    f"without measuring the new major first is the F12 defect."
                )
            return

        installed = _installed_transformers_version()
        assert installed is not None, (
            f"'{spec}' is unbounded, so the only remaining guard is that the "
            f"version this suite runs against is a measured one — but "
            f"transformers is not importable here."
        )
        running_major = installed.split(".")[0]
        assert running_major in self.MEASURED_MAJORS, (
            f"'{spec}' is unbounded, so it admits any transformers major, and "
            f"the suite is running {installed} — whose major {running_major}.x "
            f"is not in MEASURED_MAJORS. An unbounded declaration is a decision "
            f"to accept new majors as they arrive; it is not a decision to run "
            f"them unmeasured."
        )
