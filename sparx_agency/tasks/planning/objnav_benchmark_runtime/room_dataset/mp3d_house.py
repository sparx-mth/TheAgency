"""Regions (room type, box) and objects of a Matterport3D scene from its .house file. MP3D is z-up; habitat is y-up with north flipped."""


PLACEHOLDER_CATEGORIES = ("remove", "delete", "unknown", "")


def clean_category(name):
    """The annotator's category with its spaces back; placeholders and other-room walls are not objects (None)."""
    name = name.replace("#", " ").strip()
    return None if name in PLACEHOLDER_CATEGORIES or "other room" in name else name


def to_habitat(xyz):
    x, y, z = xyz
    return (x, z, -y)


def region_box_xz(region):
    """The region's box on the habitat floor plan as (x low, z low, x high, z high)."""
    return (region["lo"][0], -region["hi"][1], region["hi"][0], -region["lo"][1])


def read_house(path):
    regions, categories, objects = [], {}, []
    for line in open(path):
        t = line.split()
        if t[:1] == ["C"]:
            categories[int(t[1])] = t[3]
        elif t[:1] == ["R"]:
            regions.append({"region": int(t[1]), "level": int(t[2]), "label": t[5], "center": tuple(map(float, t[6:9])),
                            "lo": tuple(map(float, t[9:12])), "hi": tuple(map(float, t[12:15]))})
        elif t[:1] == ["O"] and int(t[2]) >= 0:
            objects.append({"region": int(t[2]), "category_index": int(t[3]), "center": tuple(map(float, t[4:7]))})
    for o in objects:
        o["category"] = clean_category(categories.get(o.pop("category_index"), ""))
    return {"regions": regions, "objects": [o for o in objects if o["category"]]}
