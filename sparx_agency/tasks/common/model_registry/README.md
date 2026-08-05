# Model / engine registry

Answers one question: *where is the TensorRT engine for this model, on this
machine?* TensorRT engines are locked to the exact GPU + TensorRT build that
produced them, so they can't be committed to git or baked into a container
image (`.gitignore` already excludes `*.engine`/`*.onnx`/etc: "downloaded/built
on the target, never committed").

## Lookup order

`resolve()` tries, in order, and raises `ArtifactMissingError` with the exact
reason for each skipped step if none succeed:

1. **Local cache** (`~/.cache/sparx_agency/models`, override with `SPARX_MODEL_CACHE`).
2. **`SPARX_MODEL_PATH`** (colon-separated, PATH-style) and other tasks'
   committed engine directories (`yolo_world_trt/engines/`, `navdp/engines/`,
   `flownav/engines/`) -- read-only search roots.
3. **`legacy_paths`** listed in the manifest for this exact variant -- lets an
   engine already sitting at its old hardcoded path resolve with zero changes.
4. **Download** from the configured store, if a prebuilt artifact is
   published for this exact device tag and TRT/arch are compatible.
5. **Build** from ONNX -- only if the caller opts in explicitly
   (`allow_build=True` / `ensure()` / `cli path --build`). A multi-minute
   build kicking off implicitly inside a running node is worse than a clear
   error.

## Usage

```python
from sparx_agency.tasks.common.model_registry.resolver import resolve

artifact = resolve("da3_metric_large", role="depth_only", precision="fp16",
                   resolution="546x364")
print(artifact.path, artifact.origin)  # origin: local|legacy|download|build
```

From a shell script (path on stdout, everything else on stderr):

```bash
ENGINE="$(python -m sparx_agency.tasks.common.model_registry.cli path \
  --model da3_metric_large --role depth_only --precision fp16 \
  --resolution 546x364)" || exit 1
```

## Files

- `key.py` -- `ModelKey`, the naming scheme (`<model>[.<role>].<precision>[.<H>x<W>].engine`).
- `manifest.py` -- reads the committed `configs/model_registry.json` (models,
  legacy paths, download sources -- never credentials or artifacts themselves).
- `paths.py` -- the local cache root and search path; refuses a cache root
  inside this repo (it may be mounted read-only, e.g. by FALCON's `run_falcon.sh`).
- `sidecar.py` -- the per-artifact `.engine.json`: which GPU/TRT build produced
  it, so a stale or wrong-device engine fails with "rebuild, here's why"
  instead of a confusing crash inside `deserialize_cuda_engine`.
- `resolver.py` -- the lookup order above.
- `cli.py` -- `path` (resolve one artifact) and `list` (show what's cached).
- `download/` -- `ArtifactStore` backends (`http.py` stdlib-only, `s3.py` lazy
  `boto3`); a new backend is one file plus a `factory.py` entry.
- `build/` -- `onnx_source.py` (find or fail loud with the export command),
  `trt_build.py` (ONNX -> engine, modeled on `yolo_world_trt/build_engine.py`).

## Adding a new model or device

Add a `variants` entry to `configs/model_registry.json` with the new
precision/resolution/role. Once an engine has been built for a new device
tag, `cli publish` (not yet wired up) would upload it and print the
`artifacts.<target_tag>` record to paste into the manifest; until then, add
it by hand with `sha256`, `bytes`, `trt_version`, and `arch`.

## No credentials, ever

The manifest stores only env-var *names* (`SPARX_MODEL_BUCKET`, ...) --
never a key or secret. The intended pattern for a container that can't hold
credentials (e.g. FALCON's read-only mount) is: the host prefetches, the
container reads the bind-mounted cache.
