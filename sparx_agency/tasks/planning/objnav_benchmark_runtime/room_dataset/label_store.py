"""The room names Daphna picks in the labelling page, kept in a CSV that is rewritten on every choice."""
import csv
import os
from pathlib import Path

ROOM_TYPES = ("bedroom", "bathroom", "toilet", "kitchen", "living room", "dining room", "family room or lounge", "tv room", "office",
              "library", "hallway", "stairs", "entryway", "closet", "laundry room", "utility room", "garage", "game room", "gym",
              "balcony or porch", "outdoor", "other room", "unsure", "not a room")


class LabelStore:
    def __init__(self, path):
        self.path = Path(path)
        self.labels = {}
        if self.path.exists():
            with open(self.path, newline="") as handle:
                self.labels = {row["room"]: row["room_type"] for row in csv.DictReader(handle)}

    def get(self, room):
        return self.labels.get(room)

    def all(self):
        return dict(self.labels)

    def set(self, room, room_type):
        if room_type not in ROOM_TYPES:
            raise ValueError("not a listed room type: %r" % room_type)
        self.labels[room] = room_type
        temporary = self.path.with_suffix(".tmp")
        with open(temporary, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["room", "room_type"])
            writer.writerows(self.labels.items())
        os.replace(temporary, self.path)
