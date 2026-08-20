"""Reference :class:`..adapter.ModelAdapter` implementations, meant to be copied.

Importing this package registers every adapter it ships, so
``adapter.available()`` lists them and ``adapter.create(name)`` builds them::

    from sparx_agency.tasks.common.trt_optimizer import adapter
    from sparx_agency.tasks.common.trt_optimizer import adapters  # noqa: F401

    classifier = adapter.check(adapter.create("image_classifier"))

Importing stays cheap: these modules pull in numpy and the pure spec types only,
and defer torch and torchvision to the inside of ``load``. That is the same rule
every adapter must follow, since a plan is routinely inspected on a machine that
cannot build anything.

The two shipped adapters are deliberately from different task families -- one
image classifier, one semantic segmenter -- because the generic pipeline's one
network-specific question, "what counts as the same answer", has a genuinely
different answer for each. See ``README.md`` beside this file for the mapping
from a task family to its decision metric.
"""
from __future__ import annotations

from sparx_agency.tasks.common.trt_optimizer.adapters.image_classifier import (
    ImageClassifierAdapter,
)
from sparx_agency.tasks.common.trt_optimizer.adapters.segmentation import (
    SemanticSegmentationAdapter,
)

__all__ = ["ImageClassifierAdapter", "SemanticSegmentationAdapter"]
