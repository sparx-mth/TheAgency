"""Which of the six ObjectNav goal categories a set of raw HM3D object names stands for, by the benchmark's own name table."""
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.hm3d_categories import PROMPT_CATEGORY, vocabulary_label


def category_of(raw_name):
    word = vocabulary_label(raw_name)
    return PROMPT_CATEGORY.get(word) if word is not None else None


def target_categories(raw_names):
    """The sorted goal categories some name stands for."""
    return sorted({c for c in map(category_of, raw_names) if c})
