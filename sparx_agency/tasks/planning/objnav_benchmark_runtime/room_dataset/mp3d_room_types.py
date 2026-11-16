"""Matterport3D's room-type letters folded into the groups a model can tell apart, and a model's free-text answer read as a group."""

ROOM_POOL = ("bathroom", "bedroom", "kitchen", "living room", "dining room", "hallway", "stairs", "entryway", "closet", "office",
             "library", "laundry or utility room", "garage", "outdoor", "recreation room", "other")
ROOM_GROUPS = ROOM_POOL

LABEL_GROUP = {"a": "bathroom", "t": "bathroom", "b": "bedroom", "k": "kitchen", "l": "living room", "f": "living room", "n": "living room",
               "v": "living room", "d": "dining room", "D": "dining room", "h": "hallway", "s": "stairs", "e": "entryway", "c": "closet",
               "o": "office", "m": "office", "i": "library", "j": "laundry or utility room", "u": "laundry or utility room", "g": "garage",
               "p": "outdoor", "x": "outdoor", "y": "outdoor", "r": "recreation room", "w": "recreation room", "B": "recreation room",
               "z": "other", "Z": "other", "C": "other", "S": "other"}

#: First match wins, so the more specific words come first (an entrance hall is an entryway, not a hallway).
ANSWER_WORDS = (("bathroom", ("bath", "restroom", "toilet", "washroom", "lavatory", "shower")),
                ("bedroom", ("bedroom", "bed room", "nursery")),
                ("kitchen", ("kitchen",)),
                ("dining room", ("dining",)),
                ("entryway", ("entry", "foyer", "lobby", "vestibule", "entrance")),
                ("living room", ("living", "family", "lounge", "sitting", "tv room", "television", "den", "great room", "parlor")),
                ("hallway", ("hallway", "hall", "corridor", "landing")),
                ("stairs", ("stair",)),
                ("closet", ("closet", "wardrobe", "dressing")),
                ("office", ("office", "study", "workspace", "meeting", "conference")),
                ("library", ("library",)),
                ("laundry or utility room", ("laundry", "utility", "mudroom", "boiler")),
                ("garage", ("garage",)),
                ("outdoor", ("balcony", "porch", "terrace", "patio", "deck", "yard", "garden", "outdoor", "driveway")),
                ("recreation room", ("game", "recreation", "rec room", "playroom", "gym", "workout", "exercise", "billiard")),
                ("other", ("basement", "attic", "other", "storage", "unknown")))


def group_of_label(letter):
    return LABEL_GROUP.get(letter)


def group_of_answer(text):
    lowered = str(text or "").lower()
    return next((group for group, words in ANSWER_WORDS if any(w in lowered for w in words)), None)
