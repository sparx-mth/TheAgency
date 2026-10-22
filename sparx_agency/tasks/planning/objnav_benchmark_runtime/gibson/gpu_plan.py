"""Gibson's import path for the shared GPU placement plan (``..gpu_plan``); also its CLI (``run_benchmark.sh``)."""
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gpu_plan import (  # noqa: F401
    COMPONENT_NAMES, DEFAULT_RESERVE_MIB, HABITAT_RENDER_MIB, LLM_OVERHEAD_FACTOR, LLM_SLOT_FACTOR, PARTIAL_FRACTION,
    PINS, YOLO_WORLD_X_MIB, Component, GpuPlan, Placement, components, llm_need_mib, main, nvidia_smi_memory,
    ollama_model_mib, pins_from_environment, plan, plan_from_environment, render_env, render_environment,
    replay_environment)

if __name__ == "__main__":
    raise SystemExit(main())
