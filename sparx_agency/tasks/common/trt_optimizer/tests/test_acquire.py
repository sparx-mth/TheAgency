"""Tests for phase 0: getting a model here and saying what arrived.

Every repository in here is built from scratch in ``tmp_path``, and none of
them is a VLA -- one is an image classifier and one is a speech encoder,
because the module must not have learned anything about navigation from the
network that happened to be on the bench when it was written. What is being
pinned is the *shape* of a released research checkout: a README, a licence, a
requirements file with the usual mess in it, some configs, a model definition
and some weights.

Three behaviours here are load-bearing rather than incidental.

* **The scan never imports.** ``model/net.py`` in the classifier fixture raises
  at module scope and writes a sentinel file. If the AST scan ever became an
  import, :func:`test_model_defs_are_found_without_importing_the_repository`
  fails on the sentinel rather than on a traceback -- which is the point, since
  a real research repo's module scope downloads weights and edits ``sys.path``
  instead of raising politely.
* **A dry run touches nothing.** ``subprocess.run`` is replaced by a stub that
  fails the test if it is called at all, so "no network work" is checked rather
  than asserted in a docstring.
* **A local source is referenced, not copied.** The operator's checkout is the
  thing analysed, so their uncommitted edit is the one that gets profiled.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.common.trt_optimizer import acquire as A


# --------------------------------------------------------------------------
# fixtures: two repositories from two different corners of the field
# --------------------------------------------------------------------------

CLASSIFIER_NET = '''"""A tiny image classifier. Importing this must never happen."""
from pathlib import Path

import torch
import torch.nn as nn

# A stranger's repo runs whatever it likes at module scope. This stands in for
# the CUDA probe / weight download / sys.path edit that real ones do.
Path(__file__).with_name("IMPORTED").write_text("the scan imported the repo")
raise RuntimeError("importing an acquired repository executes it")


class TinyResNet(nn.Module):
    def forward(self, x):
        return x


class NotAModel(object):
    pass
'''

CLASSIFIER_TRAIN = '''import argparse


def main():
    parser = argparse.ArgumentParser(description="train the classifier")
    return parser.parse_args()


if __name__ == "__main__":
    main()
'''

CLASSIFIER_REQUIREMENTS = """\
# training deps, pinned the way research repos pin them

torch>=2.0,<3.0
torchvision == 0.15.2
numpy          # no pin on purpose
pyyaml>=5.4 ; python_version >= "3.8"
timm[extras]>=0.9
-r requirements-dev.txt
--extra-index-url https://download.pytorch.org/whl/cu121
-e .
definitely-not-a-real-package>=1.0
"""

MIT_LICENSE = """\
MIT License

Copyright (c) 2024 Somebody

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
"""

NONCOMMERCIAL_LICENSE = """\
Model License

This model and its weights are released for research purposes only. Any
commercial use is prohibited without a separate written agreement.
"""

CC_BY_NC = """\
Creative Commons Attribution-NonCommercial 4.0 International Public License

By exercising the Licensed Rights, You accept and agree to be bound by these
terms. NonCommercial means not primarily intended for commercial advantage.
"""

ASR_PYPROJECT = """\
[project]
name = "speech-encoder"
version = "0.3.0"
dependencies = [
    "numpy>=1.24",
    "soundfile==0.12.1",
]

[tool.poetry.dependencies]
python = "^3.10"
librosa = "^0.10.1"
"""


def _write(path, text):
    """Create parents and write a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def classifier_repo(root):
    """An image-classification checkout: MIT, requirements.txt, one .pt."""
    _write(root / "README.md", "# tiny-resnet\n\nAn image classifier.\n")
    _write(root / "LICENSE", MIT_LICENSE)
    _write(root / "requirements.txt", CLASSIFIER_REQUIREMENTS)
    _write(root / "requirements-dev.txt", "pytest>=1.0\n")
    _write(root / "configs" / "resnet.yaml", "arch: resnet18\nclasses: 1000\n")
    _write(root / "configs" / "labels.json", '{"0": "tench"}\n')
    _write(root / "model" / "net.py", CLASSIFIER_NET)
    _write(root / "train.py", CLASSIFIER_TRAIN)
    _write(root / "broken.py", "def oops(:\n")
    _write(root / "__pycache__" / "hidden.py", "class Ghost(nn.Module): pass\n")
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "weights" / "resnet18.pt").write_bytes(b"\x00" * 4096)
    (root / "weights" / "head.onnx").write_bytes(b"\x00" * 512)
    return root


def asr_repo(root):
    """A speech-encoder checkout: pyproject, safetensors, no requirements.txt."""
    _write(root / "README.rst", "speech encoder\n")
    _write(root / "LICENSE.md", NONCOMMERCIAL_LICENSE)
    _write(root / "pyproject.toml", ASR_PYPROJECT)
    _write(root / "conformer" / "encoder.py",
           "import torch\n\n\nclass Conformer(torch.nn.Module):\n"
           "    def forward(self, wav):\n        return wav\n")
    _write(root / "conformer" / "lightning.py",
           "from pl import LightningModule\n\n\n"
           "class Trainer(LightningModule):\n    pass\n")
    (root / "conformer" / "model.safetensors").write_bytes(b"\x00" * 8192)
    return root


@pytest.fixture
def classifier(tmp_path):
    return classifier_repo(tmp_path / "checkouts" / "tiny-resnet")


@pytest.fixture
def asr(tmp_path):
    return asr_repo(tmp_path / "checkouts" / "speech-encoder")


@pytest.fixture
def no_subprocess(monkeypatch):
    """Make any subprocess call a test failure, and count the attempts."""
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a subprocess was launched: %r" % (args,))

    monkeypatch.setattr(A.subprocess, "run", forbidden)
    return calls


# --------------------------------------------------------------------------
# slugify
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source,expected", [
    ("https://github.com/pytorch/vision", "vision"),
    ("https://github.com/pytorch/vision.git", "vision"),
    ("https://github.com/pytorch/vision/", "vision"),
    ("https://github.com/pytorch/vision.git/", "vision"),
    ("http://github.com/pytorch/vision", "vision"),
    ("git@github.com:pytorch/vision.git", "vision"),
    ("ssh://git@gitlab.com/team/depth-anything.git", "depth-anything"),
    ("github.com/openai/whisper", "whisper"),
    ("https://github.com/facebookresearch/segment-anything/tree/main",
     "segment-anything"),
    ("https://github.com/ultralytics/yolov5/blob/master/models/yolo.py",
     "yolov5"),
    ("https://github.com/pytorch/vision?tab=readme-ov-file", "vision"),
    ("hf://openai/whisper-large-v3", "whisper-large-v3"),
    ("https://huggingface.co/facebook/wav2vec2-base", "wav2vec2-base"),
])
def test_slugify_reduces_every_url_form_to_the_repository_name(source, expected):
    assert A.slugify(source) == expected


@pytest.mark.parametrize("source,expected", [
    ("/home/me/checkouts/Speech Encoder", "speech-encoder"),
    ("/home/me/checkouts/Speech Encoder/", "speech-encoder"),
    ("./models/YOLO_v8.1", "yolo-v8-1"),
    ("../nn/tree/model", "model"),
])
def test_slugify_handles_a_local_path(source, expected):
    assert A.slugify(source) == expected


def test_slugify_does_not_strip_a_view_segment_from_a_local_path():
    # 'tree' is a directory name here, not GitHub's web view.
    assert A.slugify("/data/tree/resnet") == "resnet"


def test_slugify_is_bounded_in_length():
    slug = A.slugify("https://github.com/o/" + "a" * 300)
    assert len(slug) == A.MAX_SLUG_CHARS


@pytest.mark.parametrize("source", ["", "   ", "///", "https://", "!!!"])
def test_slugify_refuses_what_it_cannot_name(source):
    with pytest.raises(ValueError):
        A.slugify(source)


# --------------------------------------------------------------------------
# workspace_for
# --------------------------------------------------------------------------

def test_workspace_for_creates_the_three_working_directories(tmp_path):
    workspace = A.workspace_for("tiny-resnet", root=tmp_path / "trt")
    assert workspace == (tmp_path / "trt" / "tiny-resnet").resolve()
    for name in A.WORKSPACE_DIRS:
        assert (workspace / name).is_dir()


def test_workspace_for_honours_the_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv(A.ENV_WORKSPACE, str(tmp_path / "elsewhere"))
    workspace = A.workspace_for("whisper")
    assert workspace == (tmp_path / "elsewhere" / "whisper").resolve()
    assert (workspace / A.CODE_DIR).is_dir()


def test_an_explicit_root_outranks_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(A.ENV_WORKSPACE, str(tmp_path / "env"))
    workspace = A.workspace_for("whisper", root=tmp_path / "explicit")
    assert workspace == (tmp_path / "explicit" / "whisper").resolve()
    assert not (tmp_path / "env").exists()


def test_workspace_for_expands_a_user_relative_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    workspace = A.workspace_for("yolov8", root="~/scratch")
    assert workspace == (tmp_path / "home" / "scratch" / "yolov8").resolve()


def test_workspace_for_refuses_a_root_inside_this_repository():
    with pytest.raises(ValueError) as excinfo:
        A.workspace_for("whisper", root=A.repo_root() / "sparx_agency")
    assert "inside this repository" in str(excinfo.value)


def test_workspace_for_refuses_the_repository_root_itself():
    checkout = A.repo_root()
    with pytest.raises(ValueError):
        A.workspace_for(checkout.name, root=checkout.parent)


def test_workspace_for_refuses_a_root_inside_the_repo_from_the_environment(
        monkeypatch):
    monkeypatch.setenv(A.ENV_WORKSPACE, str(A.repo_root() / "build"))
    with pytest.raises(ValueError):
        A.workspace_for("whisper")


# --------------------------------------------------------------------------
# classify + acquire
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source,kind", [
    ("https://github.com/openai/whisper", "git"),
    ("git@github.com:openai/whisper.git", "git"),
    ("github.com/openai/whisper", "git"),
    ("hf://openai/whisper-large-v3", "hub"),
    ("https://huggingface.co/facebook/wav2vec2-base", "hub"),
])
def test_classify_reads_the_form_of_the_source(source, kind):
    assert A.classify(source) == kind


def test_classify_calls_an_existing_directory_local(classifier):
    assert A.classify(str(classifier)) == "local"


def test_classify_refuses_a_source_that_is_neither(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        A.classify(str(tmp_path / "typo-in-the-path"))
    assert "neither an existing path nor a repository URL" in str(excinfo.value)


def test_dry_run_does_no_git_work_and_leaves_nothing_behind(
        tmp_path, no_subprocess):
    source = A.acquire("https://github.com/openai/whisper.git",
                       workspace_root=tmp_path / "trt", dry_run=True)
    assert no_subprocess == []
    assert source.kind == "git"
    assert source.slug == "whisper"
    assert source.commit is None
    assert source.workspace == (tmp_path / "trt" / "whisper").resolve()
    assert not source.workspace.exists()
    assert not (tmp_path / "trt").exists()


def test_dry_run_still_reports_where_the_code_would_land(tmp_path, no_subprocess):
    source = A.acquire("https://github.com/openai/whisper",
                       workspace_root=tmp_path, dry_run=True)
    assert source.code_dir == (tmp_path / "whisper" / A.CODE_DIR).resolve()


def test_a_url_without_a_scheme_is_recorded_as_https(tmp_path, no_subprocess):
    source = A.acquire("github.com/openai/whisper", workspace_root=tmp_path,
                       dry_run=True)
    assert source.url_or_path == "https://github.com/openai/whisper"


def test_acquire_records_a_local_path_without_copying_it(
        tmp_path, classifier, no_subprocess):
    source = A.acquire(str(classifier), workspace_root=tmp_path / "trt")

    assert source.kind == "local"
    assert source.url_or_path == str(classifier.resolve())
    assert source.code_dir == classifier.resolve()
    assert source.slug == "tiny-resnet"
    # the workspace exists for notes and artifacts, but holds no copy
    assert (source.workspace / A.NOTES_DIR).is_dir()
    assert list((source.workspace / A.CODE_DIR).iterdir()) == []
    assert (classifier / "weights" / "resnet18.pt").exists()


def test_acquire_reads_the_licence_of_a_local_checkout(tmp_path, classifier):
    source = A.acquire(str(classifier), workspace_root=tmp_path)
    assert source.license_name == "MIT"


def test_acquire_records_the_commit_of_a_local_git_checkout(
        tmp_path, classifier, monkeypatch):
    (classifier / ".git").mkdir()
    seen = []

    class Done(object):
        returncode = 0
        stdout = "9f8e7d6c5b4a\n"
        stderr = ""

    def fake_run(command, **kwargs):
        seen.append(command)
        return Done()

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    source = A.acquire(str(classifier), workspace_root=tmp_path)
    assert source.commit == "9f8e7d6c5b4a"
    assert seen == [["git", "rev-parse", "HEAD"]]


def test_acquire_refuses_to_clone_over_something_that_is_not_a_checkout(
        tmp_path, no_subprocess):
    occupied = tmp_path / "trt" / "whisper" / A.CODE_DIR
    occupied.mkdir(parents=True)
    (occupied / "somebody-elses-work.txt").write_text("hello")

    with pytest.raises(RuntimeError) as excinfo:
        A.acquire("https://github.com/openai/whisper",
                  workspace_root=tmp_path / "trt")
    assert "not a git checkout" in str(excinfo.value)
    assert no_subprocess == []


def test_acquire_refuses_a_workspace_inside_this_repository(no_subprocess):
    with pytest.raises(ValueError):
        A.acquire("https://github.com/openai/whisper",
                  workspace_root=A.repo_root() / "sparx_agency", dry_run=True)


def test_git_missing_is_reported_with_what_to_do(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(A.subprocess, "run", no_git)
    with pytest.raises(RuntimeError) as excinfo:
        A.acquire("https://github.com/openai/whisper", workspace_root=tmp_path)
    assert "git is not on PATH" in str(excinfo.value)


def test_a_failed_clone_raises_with_the_git_error(tmp_path, monkeypatch):
    class Failed(object):
        returncode = 128
        stdout = ""
        stderr = "fatal: repository not found"

    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: Failed())
    with pytest.raises(RuntimeError) as excinfo:
        A.acquire("https://github.com/openai/nope", workspace_root=tmp_path)
    assert "fatal: repository not found" in str(excinfo.value)


# --------------------------------------------------------------------------
# find_entrypoints
# --------------------------------------------------------------------------

def test_model_defs_are_found_without_importing_the_repository(classifier):
    inventory = A.find_entrypoints(classifier)

    paths = [entry["path"] for entry in inventory["model_defs"]]
    assert "model/net.py" in paths
    classes = dict((e["path"], e["classes"]) for e in inventory["model_defs"])
    assert classes["model/net.py"] == ["TinyResNet"]
    assert not (classifier / "model" / "IMPORTED").exists()


def test_a_module_subclass_is_found_through_any_spelling(asr):
    classes = dict((e["path"], e["classes"])
                   for e in A.find_entrypoints(asr)["model_defs"])
    assert classes["conformer/encoder.py"] == ["Conformer"]
    assert classes["conformer/lightning.py"] == ["Trainer"]


def test_an_unparseable_file_does_not_abort_the_scan(classifier):
    inventory = A.find_entrypoints(classifier)
    assert [e["path"] for e in inventory["model_defs"]] == ["model/net.py"]


def test_cache_directories_are_not_walked(classifier):
    inventory = A.find_entrypoints(classifier)
    assert all("__pycache__" not in e["path"] for e in inventory["model_defs"])


def test_checkpoints_are_listed_with_their_sizes_largest_first(classifier):
    checkpoints = A.find_entrypoints(classifier)["checkpoints"]
    assert [c["path"] for c in checkpoints] == ["weights/resnet18.pt",
                                                "weights/head.onnx"]
    assert [c["bytes"] for c in checkpoints] == [4096, 512]


def test_every_checkpoint_suffix_is_recognised(tmp_path):
    for index, suffix in enumerate(A.CHECKPOINT_SUFFIXES):
        (tmp_path / ("weights%d%s" % (index, suffix))).write_bytes(b"x" * 10)
    found = A.find_entrypoints(tmp_path)["checkpoints"]
    assert len(found) == len(A.CHECKPOINT_SUFFIXES)


def test_the_readme_licence_configs_and_requirements_are_found(classifier):
    inventory = A.find_entrypoints(classifier)
    assert inventory["readme"] == "README.md"
    assert inventory["license"] == "LICENSE"
    assert inventory["requirements"] == ["requirements-dev.txt",
                                         "requirements.txt"]
    assert inventory["configs"] == ["configs/labels.json", "configs/resnet.yaml"]


def test_a_pyproject_counts_as_a_requirements_file(asr):
    assert A.find_entrypoints(asr)["requirements"] == ["pyproject.toml"]


def test_entrypoints_are_the_files_a_human_would_run(classifier):
    assert A.find_entrypoints(classifier)["entrypoints"] == ["train.py"]


def test_an_argparse_only_file_is_still_an_entrypoint(tmp_path):
    _write(tmp_path / "infer.py",
           "import argparse\np = argparse.ArgumentParser()\n")
    assert A.find_entrypoints(tmp_path)["entrypoints"] == ["infer.py"]


def test_a_repository_with_none_of_it_reports_empty_rather_than_guessing(
        tmp_path):
    (tmp_path / "empty").mkdir()
    inventory = A.find_entrypoints(tmp_path / "empty")
    assert inventory["readme"] is None
    assert inventory["license"] is None
    assert inventory["checkpoints"] == []
    assert inventory["model_defs"] == []


def test_find_entrypoints_refuses_a_path_that_is_not_a_directory(tmp_path):
    with pytest.raises(ValueError):
        A.find_entrypoints(tmp_path / "never-cloned")


# --------------------------------------------------------------------------
# requirements_report
# --------------------------------------------------------------------------

def _rows(code_dir):
    return dict((name, (spec, installed, ok))
                for (name, spec, installed, ok)
                in A.requirements_report(code_dir))


def test_requirements_are_parsed_with_their_specifiers(classifier):
    rows = _rows(classifier)
    assert rows["torch"][0] == ">=2.0,<3.0"
    assert rows["torchvision"][0] == "== 0.15.2"
    assert rows["numpy"][0] == ""
    assert rows["pyyaml"][0] == ">=5.4"
    assert rows["timm"][0] == ">=0.9"


def test_comments_blank_lines_and_pip_options_are_dropped(classifier):
    names = [row[0] for row in A.requirements_report(classifier)]
    assert "" not in names
    assert not any(name.startswith("-") for name in names)
    assert not any("http" in name for name in names)


def test_an_installed_package_is_reported_as_satisfied(classifier):
    # requirements-dev.txt asks for pytest>=1.0, and pytest is running.
    spec, installed, satisfied = _rows(classifier)["pytest"]
    assert spec == ">=1.0"
    assert installed is not None
    assert satisfied is True


def test_an_absent_package_is_reported_as_missing_not_satisfied(classifier):
    spec, installed, satisfied = _rows(classifier)["definitely-not-a-real-package"]
    assert installed is None
    assert satisfied is False


def test_an_unmeetable_pin_on_an_installed_package_fails(tmp_path):
    _write(tmp_path / "requirements.txt", "pytest>=99999\n")
    spec, installed, satisfied = _rows(tmp_path)["pytest"]
    assert installed is not None
    assert satisfied is False


def test_pep621_dependencies_are_read_from_a_pyproject(asr):
    rows = _rows(asr)
    assert rows["numpy"][0] == ">=1.24"
    assert rows["soundfile"][0] == "==0.12.1"


def test_a_poetry_table_is_read_too_and_a_caret_is_left_undecided(asr):
    rows = _rows(asr)
    assert rows["librosa"][0] == "^0.10.1"
    assert rows["librosa"][2] is None or rows["librosa"][2] is False


def test_a_url_pin_is_reported_verbatim(tmp_path):
    _write(tmp_path / "requirements.txt",
           "somepkg @ git+https://github.com/o/somepkg@abc123\n")
    spec, installed, satisfied = _rows(tmp_path)["somepkg"]
    assert spec.startswith("@ git+https://")
    assert installed is None
    assert satisfied is False


def test_requirements_report_installs_nothing(classifier, no_subprocess):
    A.requirements_report(classifier)
    assert no_subprocess == []


def test_requirements_report_refuses_a_missing_directory(tmp_path):
    with pytest.raises(ValueError):
        A.requirements_report(tmp_path / "never-cloned")


# --------------------------------------------------------------------------
# license_note
# --------------------------------------------------------------------------

def test_an_mit_licence_is_identified_and_not_flagged(classifier):
    note = A.license_note(classifier)
    assert note.name == "MIT"
    assert note.restricted is False
    assert note.path == "LICENSE"


def test_a_research_only_licence_is_flagged(asr):
    note = A.license_note(asr)
    assert note.restricted is True
    assert "RESTRICTED" in note.line
    assert "research purposes only" in note.line


def test_a_creative_commons_noncommercial_licence_is_flagged(tmp_path):
    _write(tmp_path / "LICENSE", CC_BY_NC)
    note = A.license_note(tmp_path)
    assert note.name == "CC-BY-NC-4.0"
    assert note.restricted is True


@pytest.mark.parametrize("filename", ["LICENSE", "LICENSE.txt", "LICENCE",
                                      "COPYING", "LICENSE.md"])
def test_the_licence_is_found_under_any_of_its_usual_names(tmp_path, filename):
    _write(tmp_path / filename, MIT_LICENSE)
    assert A.license_note(tmp_path).name == "MIT"


@pytest.mark.parametrize("text,expected", [
    ("Apache License\nVersion 2.0, January 2004\n", "Apache-2.0"),
    ("GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n", "GPL-3.0"),
    ("Redistribution and use in source and binary forms, with or without\n"
     "modification, are permitted. Neither the name of the copyright holder\n",
     "BSD-3-Clause"),
    ("Redistribution and use in source and binary forms are permitted.\n",
     "BSD-2-Clause"),
    ("Mozilla Public License Version 2.0\n", "MPL-2.0"),
    ("This is free and unencumbered software released into the public domain.",
     "Unlicense"),
])
def test_the_common_licences_are_identified(tmp_path, text, expected):
    _write(tmp_path / "LICENSE", text)
    assert A.license_note(tmp_path).name == expected


def test_a_missing_licence_is_loud_but_is_not_the_noncommercial_flag(tmp_path):
    (tmp_path / "bare").mkdir()
    note = A.license_note(tmp_path / "bare")
    assert note.name is None
    assert note.restricted is False
    assert "all rights reserved" in note.line


def test_an_unrecognised_licence_says_so_instead_of_guessing(tmp_path):
    _write(tmp_path / "LICENSE", "Do what you feel is right, roughly.\n")
    note = A.license_note(tmp_path)
    assert note.name is None
    assert "unidentified" in note.line


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------

def test_the_summary_names_the_source_and_what_was_found(tmp_path, classifier):
    source = A.acquire(str(classifier), workspace_root=tmp_path)
    text = A.summarize(source, A.find_entrypoints(source.code_dir))

    assert "tiny-resnet" in text
    assert str(source.workspace) in text
    assert "weights/resnet18.pt" in text
    assert "4 KiB" in text
    assert "model/net.py" in text and "TinyResNet" in text
    assert "MIT" in text


def test_the_summary_states_what_was_deliberately_not_done(tmp_path, classifier):
    source = A.acquire(str(classifier), workspace_root=tmp_path)
    text = A.summarize(source, A.find_entrypoints(source.code_dir))
    assert "nothing was installed" in text
    assert "no weights were downloaded" in text
    assert "imported or executed" in text


def test_the_summary_shouts_about_a_restricted_licence(tmp_path, asr):
    source = A.acquire(str(asr), workspace_root=tmp_path)
    text = A.summarize(source, A.find_entrypoints(source.code_dir))
    assert "RESTRICTED LICENCE" in text


def test_the_summary_survives_a_dry_run_with_no_code_on_disk(
        tmp_path, no_subprocess):
    source = A.acquire("https://github.com/openai/whisper",
                       workspace_root=tmp_path, dry_run=True)
    empty = dict(readme=None, license=None, requirements=[], configs=[],
                 model_defs=[], checkpoints=[], entrypoints=[])
    text = A.summarize(source, empty)
    assert "(not present)" in text
    assert "nothing can be profiled yet" in text


def test_the_summary_truncates_a_long_list_honestly(tmp_path):
    for index in range(A.SUMMARY_LIMIT + 5):
        _write(tmp_path / ("cfg%02d.yaml" % index), "a: 1\n")
    source = A.Source(kind="local", url_or_path=str(tmp_path), slug="many",
                      workspace=tmp_path)
    text = A.summarize(source, A.find_entrypoints(tmp_path))
    assert "(+5 more)" in text


# --------------------------------------------------------------------------
# the real git path, against a local repository (no network)
# --------------------------------------------------------------------------

def _init_repo(path):
    """Make a real one-commit git repository, or skip if git is unusable."""
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    (path / "model.py").write_text(
        "import torch.nn as nn\n\n\nclass Segmenter(nn.Module):\n    pass\n")
    (path / "LICENSE").write_text(MIT_LICENSE)
    commands = [
        ["git", "init", "-q", str(path)],
        ["git", "-C", str(path), "add", "."],
        ["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "-m", "initial"],
    ]
    for command in commands:
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            pytest.skip("git unusable here: %s" % (done.stderr.strip(),))


def test_a_real_shallow_clone_lands_in_the_workspace(tmp_path):
    upstream = tmp_path / "upstream" / "segmenter"
    _init_repo(upstream)

    source = A.acquire("file://%s" % upstream, workspace_root=tmp_path / "trt")

    assert source.kind == "git"
    assert source.slug == "segmenter"
    assert (source.code_dir / "model.py").is_file()
    assert len(source.commit or "") == 40
    assert source.license_name == "MIT"
    inventory = A.find_entrypoints(source.code_dir)
    assert [e["classes"] for e in inventory["model_defs"]] == [["Segmenter"]]


def test_re_acquiring_the_same_repository_reuses_the_clone(tmp_path):
    upstream = tmp_path / "upstream" / "segmenter"
    _init_repo(upstream)
    root = tmp_path / "trt"

    first = A.acquire("file://%s" % upstream, workspace_root=root)
    (first.code_dir / "LOCAL_EDIT").write_text("kept")
    second = A.acquire("file://%s" % upstream, workspace_root=root)

    assert second.commit == first.commit
    assert (second.code_dir / "LOCAL_EDIT").is_file()


def test_a_second_repository_with_the_same_name_is_refused(tmp_path):
    upstream = tmp_path / "upstream" / "segmenter"
    other = tmp_path / "elsewhere" / "segmenter"
    _init_repo(upstream)
    _init_repo(other)
    root = tmp_path / "trt"

    A.acquire("file://%s" % upstream, workspace_root=root)
    with pytest.raises(RuntimeError) as excinfo:
        A.acquire("file://%s" % other, workspace_root=root)
    assert "slugified to the same workspace name" in str(excinfo.value)
