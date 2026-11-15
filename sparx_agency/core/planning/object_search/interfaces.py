"""The seam between the object-search method and the services that perceive and reason for it.

The method is a :class:`~sparx_agency.core.planning.objnav.interfaces.policy.SearchPolicy`.
It never talks to a network, a model or a simulator itself: an **object
detector**, an optional **box-prompted segmenter** and a **room-reasoning
LLM** are handed to :class:`~sparx_agency.core.planning.object_search.policy.RPTSearchPolicy`
at construction, and this module says exactly what the method asks of each.
The HTTP clients that satisfy these contracts live with the benchmark
runtime (``tasks/planning/objnav_benchmark_runtime/services/``), because the
wire formats they speak are owned by the serving side
(``tasks/mapping/scene_graph/serve/``) and ``core`` imports nothing from
``tasks``.

**What a detection is.** :class:`Detection` is the detector's word on one
box: a class label, a confidence and a float pixel box. It is deliberately
NOT :class:`~sparx_agency.core.common.types.perception.Detection2D` -- that
record carries the frame size and an *integer* box for the object-approach
servo, and rounding the box would move every projected centroid of this
method by up to half a pixel. The fields and their order are the detection
server's wire record (``{"cls", "conf", "xyxy"}``), so a recorded run replays
through the same code without a translation step; the HTTP client converts
the wire record into this one.

**How a service fails.** Two families, told apart by the perception cycle:

- *Identity drift* -- another vocabulary, checkpoint or configuration
  answering mid-run -- is an
  :class:`~sparx_agency.core.planning.objnav.errors.ObjNavInternalError`
  raised **without a recoverable cause**. It ends the run: a back-off would
  fly on under a different detector.
- *A transport or one-frame failure* -- a timeout, a dead service, a
  malformed reply for one of half a million frames -- is the same error
  class raised ``from`` its cause: the transport library's error (an
  :class:`OSError`; ``requests.RequestException`` is one), a
  :class:`DetectorFrameError` / :class:`SegmenterFrameError`, a
  :class:`ValueError` or a :class:`KeyError`. The cycle records it on the
  exploration fallback, which arms a doubling back-off, and the search keeps
  moving instead of waiting a timeout per action.

Any other exception a service raises is treated as a frame failure too, so
a fake detector in a test that raises ``RuntimeError`` exercises the
back-off, never a dead episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from sparx_agency.core.common.math.bbox import BBox

try:  # Python 3.8 has typing.Protocol; the FALCON container never imports this module
    from typing import Protocol
except ImportError:  # pragma: no cover - Python < 3.8
    Protocol = object  # type: ignore[assignment,misc]



@dataclass(frozen=True)
class Detection:
    """One detector output: the open-vocabulary prompt that matched, its confidence and its pixel box.

    Attributes:
        cls: Class label -- the detector prompt, in the detector's own
            vocabulary (alias folding, ``couch -> sofa``, happens later in
            :func:`~sparx_agency.core.mapping.objects.detection_dedup.deduplicate_detections`).
        conf: Confidence in ``[0, 1]``.
        xyxy: Pixel bounding box ``(x1, y1, x2, y2)``, origin top-left, floats
            (:data:`~sparx_agency.core.common.math.bbox.BBox`).
    """

    cls: str
    conf: float
    xyxy: BBox


class DetectorFrameError(ValueError):
    """One frame's detector reply was unusable (shape, hash, a malformed box): the frame fails, the run goes on.

    Distinct from the identity drift a client raises as a bare
    ``ObjNavInternalError`` (another vocabulary, model or configuration
    answering), which ends the run: a malformed box in one of half a million
    frames is a frame failure for the detector back-off, not a reason to
    abandon a thousand episodes (review of 2026-10-07).
    """


class SegmenterFrameError(ValueError):
    """One frame's segmenter reply was unusable (shape, hash, a malformed mask): the frame fails, the run goes on."""


#: The causes behind an ``ObjNavInternalError`` that mark a transport or frame failure
#: (recoverable: the perception cycle backs off) rather than identity drift (fatal).
#: ``OSError`` covers the HTTP client's ``requests.RequestException`` without this
#: package importing the transport library.
RECOVERABLE_DETECTOR_CAUSES = (OSError, DetectorFrameError, ValueError, KeyError)
RECOVERABLE_SEGMENTER_CAUSES = (OSError, SegmenterFrameError, ValueError, KeyError, TypeError)


class Detector(Protocol):
    """An open-vocabulary object detector, one synchronous call per RGB frame.

    ``detect`` returns the frame's boxes as :class:`Detection` records (or
    any object with the same three attributes). The optional
    ``last_diagnostics`` mapping -- the service's own account of the frame
    (its verification mode, the request hash) -- is copied into the
    perception record when present; the method reads nothing else.
    """

    def detect(self, rgb: np.ndarray) -> Sequence[Detection]:
        """Detections for one ``HxWx3`` uint8 RGB frame."""
        ...


class Segmenter(Protocol):
    """A box-prompted instance segmenter: one boolean ``HxW`` mask per box, in box order.

    The method prompts only the target's boxes (their depth ends an
    episode). ``last_inference_ms``, when present, is recorded with the
    frame's segmenter evidence.
    """

    def segment(self, rgb: np.ndarray, boxes_xyxy: Sequence[BBox]) -> List[np.ndarray]:
        """Masks for ``boxes_xyxy`` on one ``HxWx3`` uint8 RGB frame, one per box, in order."""
        ...


class RoomReasoner(Protocol):
    """The LLM behind the room labels and the node oracle: a JSON chat completion.

    The same interface :class:`~sparx_agency.core.mapping.topology.llm_client.LLMClient`
    offers and :class:`~sparx_agency.core.mapping.topology.search_node_oracle.SearchNodeOracle`
    consumes; ``reasoning=True`` routes the one judgement per loop point to
    the larger model when the client supports the keyword.
    """

    def chat_json(self, system: str, user: str, **kwargs: Any) -> Dict[str, Any]:
        """One completion, parsed as a JSON object."""
        ...


def detection_from_wire(record: Any) -> Detection:
    """A :class:`Detection` from any record with ``cls``, ``conf`` and ``xyxy`` attributes (the HTTP wire record).

    Values are copied, never rounded: the box stays the float box the
    detector emitted.
    """
    return Detection(str(record.cls), float(record.conf), tuple(float(v) for v in record.xyxy))
