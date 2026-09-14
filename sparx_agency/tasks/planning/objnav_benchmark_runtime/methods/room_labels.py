"""Evidence-gated, revisable room labels around the existing core classifier."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from sparx_agency.core.mapping.topology.room_classifier import RoomLabel, RoomTypeClassifier


@dataclass(frozen=True)
class RoomLabelSettings:
    min_objects: int = 3
    min_classes: int = 2
    min_evidence_updates: int = 2
    refresh_steps: int = 50


class RevisableRoomLabels:
    """Reconsider labels on new evidence, periodic refresh and room splits/merges.

    Counts refer to confirmed unique landmarks, not repeated video frames.
    Door geometry is not a room-type feature. Labels stay provisional until
    repeated classification agrees, but can always be revised afterwards.
    """

    def __init__(self, client, settings=None):
        self.settings = settings or RoomLabelSettings()
        s = self.settings
        self.classifier = RoomTypeClassifier(client, min_objects=s.min_objects,
                                             min_classes=s.min_classes, count_sensitive=True)
        self.labels = {}
        self.metadata = {}
        self.history = []
        self._evidence_updates = {}
        self._signatures = {}
        self._last_query = {}
        self._agreement = {}
        self.queries = 0

    def update(self, objects, step, partition_changed=False):
        if partition_changed:
            # Pids survive one side of a split: retaining their old cached
            # label would attach the whole-room verdict to a different region.
            self._signatures.clear()
            self._evidence_updates.clear()
            self._last_query.clear()
            self._agreement.clear()
        live = set(objects)
        self.labels = {pid: label for pid, label in self.labels.items() if pid in live}
        self.metadata = {pid: item for pid, item in self.metadata.items() if pid in live}
        for pid, classes in objects.items():
            self._update_room(pid, classes, step)
        return dict(self.labels)

    def _update_room(self, pid, classes, step):
        # Prompt aliases must not count as extra kinds of semantic evidence.
        aliases = {"couch": "sofa", "tv": "television"}
        classes = [aliases.get(name, name) for name in classes]
        signature = tuple(sorted(Counter(classes).items()))
        self._evidence_updates[pid] = self._evidence_updates.get(pid, 0) + 1
        s = self.settings
        ready = (len(classes) >= s.min_objects and len(set(classes)) >= s.min_classes
                 and self._evidence_updates[pid] >= s.min_evidence_updates)
        previous = self.labels.get(pid)
        elapsed = step - self._last_query.get(pid, -s.refresh_steps)
        changed = self._signatures.get(pid) != signature
        if not ready:
            result = RoomLabel("unknown", 0.0, "insufficient stable confirmed-object evidence")
            reason = "evidence_gate"
            self._agreement[pid] = 0
        elif changed or elapsed >= s.refresh_steps:
            result = self.classifier.classify(classes, refresh=True)
            self.queries += 1
            self._last_query[pid] = step
            self._signatures[pid] = signature
            self._agreement[pid] = (self._agreement.get(pid, 0) + 1
                                   if previous and previous.label == result.label else 1)
            reason = "evidence_changed" if changed else "periodic_refresh"
        else:
            return
        self.labels[pid] = result
        if previous is None or previous.label != result.label:
            self.history.append({"step": int(step), "room_id": int(pid),
                                 "previous": previous.label if previous else None,
                                 "label": result.label, "trigger": reason,
                                 "objects": dict(signature)})
        self.metadata[pid] = dict(asdict(result), provisional=(result.label == "unknown"
                                  or result.confidence < 0.65 or self._agreement.get(pid, 0) < 2),
                                  evidence_objects=len(classes), evidence_classes=len(set(classes)),
                                  updated_step=int(step), trigger=reason)


