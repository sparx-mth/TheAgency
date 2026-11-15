"""GPU placement plan: priorities, the GPU-only reservation, partial LLM offload, the environment it emits."""
from __future__ import annotations

import json

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime import gpu_plan
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gpu_plan import (
    Component, components, llm_need_mib, nvidia_smi_memory, ollama_model_mib, plan, plan_from_environment,
    render_env)


def parts(llm=14000, yolo=2048, habitat=1536, habitat_cpu=False):
    return components(llm, "test", yolo, habitat, habitat_cpu_render=habitat_cpu)


def devices(result):
    return {item.component: item.device for item in result.placements}


def test_everything_fits_on_a_large_card_and_the_run_is_marked_shared():
    result = plan(parts(), 0, "RTX 4090", 24564, 850)
    assert devices(result) == {"llm": "gpu", "yolo": "gpu", "habitat": "gpu"}
    assert result.free_mib == 24564 - 850 - 512 and result.shared and not result.warnings
    env = result.environment()
    assert env["OLLAMA_CUDA_VISIBLE_DEVICES"] == "0" and env["LLM_NUM_GPU"] == "-1"
    assert env["DETECTOR_DEVICE"] == "cuda:0" and env["DETECTOR_CUDA_VISIBLE_DEVICES"] == "0"
    assert env["HABITAT_SOFTWARE_RENDER"] == "0" and env["ALLOW_SHARED_GPU"] == "1"
    assert [item.component for item in result.placements] == ["llm", "yolo", "habitat"], "reported in component order"


def test_a_12_gb_card_loads_the_llm_partially_and_offloads_yolo():
    """Habitat (GPU-only) is reserved first; the LLM's estimate fits more than half-way, so Ollama splits it;
    YOLO, the next least bottlenecking component, goes to the CPU."""
    result = plan(parts(), 0, "RTX 3080 Ti", 12288, 300)
    assert devices(result) == {"llm": "gpu-partial", "yolo": "cpu", "habitat": "gpu"}
    llm = result.placement("llm")
    assert llm.granted_mib == 12288 - 300 - 512 - 1536 and "partial" in llm.reason
    env = result.environment()
    assert env["LLM_NUM_GPU"] == "" and env["OLLAMA_CUDA_VISIBLE_DEVICES"] == "0"
    assert env["DETECTOR_DEVICE"] == "cpu" and env["DETECTOR_CUDA_VISIBLE_DEVICES"] == ""


def test_an_8_gb_card_cannot_hold_half_the_llm_so_yolo_takes_what_habitat_leaves():
    result = plan(parts(), 0, "RTX 3070", 8192, 300)
    assert devices(result) == {"llm": "cpu", "yolo": "gpu", "habitat": "gpu"}
    assert result.placement("llm").reason.startswith("offloaded")


def test_a_card_too_small_for_half_the_llm_sends_it_to_the_cpu_and_yolo_takes_the_card():
    result = plan(parts(llm=14000), 0, "small", 6144, 200)
    assert devices(result) == {"llm": "cpu", "yolo": "gpu", "habitat": "gpu"}
    env = result.environment()
    assert env["OLLAMA_CUDA_VISIBLE_DEVICES"] == "-1" and env["LLM_NUM_GPU"] == "0"
    assert env["DETECTOR_DEVICE"] == "cuda:0"


def test_priority_order_holds_among_cpu_capable_components():
    """Room for the LLM or YOLO but not both: the LLM (priority 1) gets it."""
    result = plan(parts(llm=4000, yolo=2048, habitat=1536), 0, "card", 7000, 0)
    assert devices(result) == {"llm": "gpu", "yolo": "cpu", "habitat": "gpu"}


def test_a_gpu_only_component_that_cannot_fit_is_placed_with_a_warning_not_dropped():
    result = plan(parts(llm=1000, yolo=1000, habitat=5000), 0, "tiny", 4096, 0)
    assert devices(result)["habitat"] == "gpu" and any("habitat needs" in w for w in result.warnings)


def test_habitat_may_be_offloaded_only_when_the_operator_allows_software_rendering():
    result = plan(parts(llm=3000, yolo=1900, habitat=1536, habitat_cpu=True), 0, "card", 5500, 0)
    assert devices(result) == {"llm": "gpu", "yolo": "gpu", "habitat": "cpu"}, "lowest priority once it is CPU-capable"
    assert result.environment()["HABITAT_SOFTWARE_RENDER"] == "1"
    assert any("software EGL" in w for w in result.warnings)


def test_a_cpu_pin_takes_nothing_off_the_budget_and_pinning_habitat_is_the_software_rendering_opt_in():
    """The 8 GB laptop, 2026-10-13: the 3B LLM on the card, the renderer on the CPU by the operator's word, YOLO
    wherever the VRAM left allows -- here it does; without the pin Habitat's 1536 MiB would have been counted."""
    pinned = components(2800, "LLM_VRAM_MIB override", 2048, 1536, pins={"llm": "gpu", "habitat": "cpu"})
    result = plan(pinned, 0, "RTX 5070 Laptop", 8151, 80)
    assert devices(result) == {"llm": "gpu", "yolo": "gpu", "habitat": "cpu"}
    assert result.pins == {"llm": "gpu", "yolo": "auto", "habitat": "cpu"}
    assert "GPU_PLAN_HABITAT=cpu" in result.placement("habitat").reason and result.placement("habitat").granted_mib == 0
    assert "GPU_PLAN_LLM=gpu" in result.placement("llm").reason
    assert result.placement("yolo").granted_mib == 2048 and "pinned" not in result.placement("yolo").reason
    env = result.environment()
    assert env["HABITAT_SOFTWARE_RENDER"] == "1" and env["DETECTOR_DEVICE"] == "cuda:0" and env["LLM_NUM_GPU"] == "-1"
    assert result.shared, "two of ours on the card"
    # YOLO still goes by the budget: on a card with room for the LLM alone it is offloaded, the pins untouched
    small = plan(pinned, 0, "card", 4000, 0)
    assert devices(small) == {"llm": "gpu", "yolo": "cpu", "habitat": "cpu"}


def test_a_gpu_pin_is_served_before_the_auto_components_and_warns_when_it_cannot_fit():
    """Pinning YOLO to the card puts it ahead of the (auto) LLM: the LLM then takes the partial path. A pin the
    budget cannot honour is placed anyway, with a warning naming the way out -- as a GPU-only component is."""
    result = plan(components(4000, "test", 2048, 1536, pins={"yolo": "gpu"}), 0, "card", 7000, 0)
    assert devices(result) == {"llm": "gpu-partial", "yolo": "gpu", "habitat": "gpu"}
    too_big = plan(components(1000, "test", 9000, 1536, pins={"yolo": "gpu"}), 0, "card", 4096, 0)
    assert devices(too_big)["yolo"] == "gpu" and "pinned to the GPU" in too_big.placement("yolo").reason
    assert any("GPU_PLAN_YOLO=auto" in w for w in too_big.warnings)


def test_pins_are_read_from_the_environment_and_validated(tmp_path):
    env = {"LLM_VRAM_MIB": "2800", "GPU_PLAN_LLM": "GPU", "GPU_PLAN_HABITAT": " cpu "}
    result = plan_from_environment(0, run=lambda cmd: "card, 8151, 80\n", env=env)
    assert devices(result) == {"llm": "gpu", "yolo": "gpu", "habitat": "cpu"}, "case and whitespace are forgiven"
    assert gpu_plan.pins_from_environment({}) == {"llm": "auto", "yolo": "auto", "habitat": "auto"}
    with pytest.raises(ValueError, match="GPU_PLAN_YOLO must be one of auto/gpu/cpu"):
        plan_from_environment(0, run=lambda cmd: "card, 8151, 80\n", env={"LLM_VRAM_MIB": "2800", "GPU_PLAN_YOLO": "maybe"})
    with pytest.raises(ValueError, match="No such component"):
        components(1000, "test", pins={"renderer": "cpu"})
    with pytest.raises(ValueError, match="cannot run there"):
        Component("habitat", 3, 1536, cpu_capable=False, pin="cpu")


def test_an_occupied_card_marks_the_run_shared_even_with_one_component_on_it():
    result = plan(parts(llm=20000, yolo=5000, habitat=1536), 0, "card", 8192, 2000)
    assert devices(result) == {"llm": "cpu", "yolo": "cpu", "habitat": "gpu"}
    assert result.shared, "2000 MiB already in use: the gibson.run gate needs --allow-shared-gpu"


def test_nvidia_smi_output_is_parsed_and_errors_surface():
    assert nvidia_smi_memory(0, lambda cmd: "NVIDIA GeForce RTX 4090, 24564, 862\n") == ("NVIDIA GeForce RTX 4090", 24564, 862)
    with pytest.raises(RuntimeError):
        nvidia_smi_memory(0, lambda cmd: "garbage\n")


def test_ollama_manifest_sizes_and_the_llm_estimate(tmp_path):
    base = tmp_path / "manifests" / "registry.ollama.ai" / "library"
    for name, size in (("qwen2.5/3b-instruct", 1_900_000_000), ("qwen2.5/14b-instruct", 9_000_000_000)):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"layers": [{"size": size}, {"size": 1000}]}))
    assert ollama_model_mib(tmp_path, "qwen2.5:14b-instruct") == round((9_000_000_000 + 1000) / 2 ** 20)
    assert ollama_model_mib(tmp_path, "qwen2.5:7b-instruct") is None
    need, note = llm_need_mib(["qwen2.5:3b-instruct", "qwen2.5:14b-instruct"], tmp_path)
    assert need == round((ollama_model_mib(tmp_path, "qwen2.5:3b-instruct") + ollama_model_mib(tmp_path, "qwen2.5:14b-instruct"))
                         * gpu_plan.LLM_OVERHEAD_FACTOR) and "x %.2f" % gpu_plan.LLM_OVERHEAD_FACTOR in note
    assert llm_need_mib(["any"], tmp_path, override=4321) == (4321, "LLM_VRAM_MIB override")
    with pytest.raises(RuntimeError, match="LLM_VRAM_MIB"):
        llm_need_mib(["qwen2.5:7b-instruct"], tmp_path)


def test_plan_from_environment_reads_models_dir_overrides_and_reserve(tmp_path):
    base = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "m"
    base.mkdir(parents=True)
    (base / "small").write_text(json.dumps({"layers": [{"size": 1_000_000_000}]}))
    env = {"LLM_MODEL": "m:small", "LLM_REASONING_MODEL": "m:small", "OLLAMA_MODELS_DIR": str(tmp_path),
           "YOLO_VRAM_MIB": "1000", "HABITAT_VRAM_MIB": "500", "GPU_RESERVE_MIB": "100"}
    result = plan_from_environment(0, run=lambda cmd: "card, 4000, 100\n", env=env)
    assert result.reserve_mib == 100 and result.free_mib == 3800
    assert result.placement("yolo").need_mib == 1000 and result.placement("habitat").need_mib == 500
    assert result.placement("llm").need_mib == round(round(1_000_000_000 / 2 ** 20) * gpu_plan.LLM_OVERHEAD_FACTOR)
    text = render_env(result)
    assert "export ALLOW_SHARED_GPU='1'" in text and "export DETECTOR_DEVICE='cuda:0'" in text


def test_component_validation_is_explicit():
    with pytest.raises(TypeError):
        Component("x", 1)   # need_mib and cpu_capable are required


def test_each_ollama_request_slot_beyond_the_first_adds_its_own_kv_cache_to_the_llm_estimate(tmp_path):
    base = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "m"
    base.mkdir(parents=True)
    (base / "big").write_text(json.dumps({"layers": [{"size": 8572 * 2 ** 20}]}))
    one, note_one = llm_need_mib(["m:big"], tmp_path)
    two, note_two = llm_need_mib(["m:big"], tmp_path, parallel=2)
    assert one == round(8572 * gpu_plan.LLM_OVERHEAD_FACTOR) and "slots" not in note_one
    assert two == round(8572 * (gpu_plan.LLM_OVERHEAD_FACTOR + gpu_plan.LLM_SLOT_FACTOR)) and "2 request slots" in note_two
    assert llm_need_mib(["m:big"], tmp_path, override=100, parallel=4) == (100, "LLM_VRAM_MIB override")
    with pytest.raises(ValueError, match="OLLAMA_NUM_PARALLEL"):
        llm_need_mib(["m:big"], tmp_path, parallel=0)


def test_parallel_shards_take_one_renderer_each_off_the_budget():
    three = components(14000, "test", habitat_count=3)
    habitat = [part for part in three if part.name == "habitat"][0]
    assert habitat.need_mib == 3 * gpu_plan.HABITAT_RENDER_MIB and "3 renderers" in habitat.note
    # 20 GB free: the LLM and YOLO fit beside three renderers but not beside six.
    assert devices(plan(three, 0, "card", 24000, 1000))["yolo"] == "gpu"
    six = components(14000, "test", habitat_count=6)
    assert devices(plan(six, 0, "card", 24000, 1000))["yolo"] == "cpu"
    with pytest.raises(ValueError, match="HABITAT_RENDERERS"):
        components(14000, "test", habitat_count=0)


def test_plan_from_environment_scales_by_slots_and_renderers(tmp_path):
    base = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "m"
    base.mkdir(parents=True)
    (base / "small").write_text(json.dumps({"layers": [{"size": 1000 * 2 ** 20}]}))
    env = {"LLM_MODEL": "m:small", "LLM_REASONING_MODEL": "m:small", "OLLAMA_MODELS_DIR": str(tmp_path),
           "OLLAMA_NUM_PARALLEL": "3", "HABITAT_RENDERERS": "4", "HABITAT_VRAM_MIB": "500"}
    result = plan_from_environment(0, run=lambda cmd: "card, 24000, 100\n", env=env)
    assert result.placement("llm").need_mib == round(1000 * (gpu_plan.LLM_OVERHEAD_FACTOR + 2 * gpu_plan.LLM_SLOT_FACTOR))
    assert result.placement("habitat").need_mib == 2000


def test_a_recorded_plan_is_replayed_as_the_environment_it_emitted(tmp_path):
    result = plan(parts(), 0, "card", 24000, 1000)
    record = tmp_path / "gpu_plan.json"
    record.write_text(json.dumps(dict(result.__dict__, environment=result.environment()), default=lambda o: o.__dict__))
    assert gpu_plan.replay_environment(record) == result.environment()
    copy = tmp_path / "elsewhere" / "gpu_plan.json"
    assert gpu_plan.main(["--replay", str(record), "--format", "env", "--write", str(copy)]) == 0
    assert copy.read_text() == record.read_text()
    (tmp_path / "bare.json").write_text(json.dumps({"placements": []}))
    with pytest.raises(RuntimeError, match="environment"):
        gpu_plan.replay_environment(tmp_path / "bare.json")
    with pytest.raises(RuntimeError):
        gpu_plan.replay_environment(tmp_path / "missing.json")
