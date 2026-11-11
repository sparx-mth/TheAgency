"""What counts as each kind of object: the annotator's words on one side, the detector's labels on the other."""

#: kind -> substrings of the HM3D annotation names that are that kind, and the ones that look like it but are not.
GT_WORDS = {
    "bed": (("bed", "crib"), ("bedside", "bed table")),
    "sofa": (("sofa", "couch", "loveseat"), ()),
    "armchair": (("armchair",), ()),
    "chair": (("chair",), ("armchair",)),
    "tv": (("tv", "television", "monitor"), ("stand",)),
    "toilet": (("toilet",), ("paper", "brush", "bin")),
    "plant": (("plant",), ()),
    "sink": (("sink", "washbasin"), ()),
    "oven": (("oven", "stove", "cooktop"), ()),
    "refrigerator": (("refrigerator", "fridge"), ()),
    "bathtub": (("bath tub", "bathtub"), ()),
    "desk": (("desk",), ()),
    "dining table": (("dining table",), ()),
}

#: the detector's labels (the benchmark vocabulary) -> the kind each answers
DETECTOR_KIND = {
    "bed": "bed", "sofa": "sofa", "couch": "sofa", "armchair": "armchair", "chair": "chair", "tv": "tv", "television": "tv",
    "monitor": "tv", "toilet": "toilet", "potted plant": "plant", "sink": "sink", "oven": "oven", "refrigerator": "refrigerator",
    "bathtub": "bathtub", "desk": "desk", "dining table": "dining table",
}


def kind_of_annotation(name):
    """The kind of an annotation name, or None when it is none of the kinds."""
    lowered = str(name).lower()
    for kind, (words, vetoes) in GT_WORDS.items():
        if any(w in lowered for w in words) and not any(v in lowered for v in vetoes):
            return kind
    return None
