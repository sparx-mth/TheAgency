"""Ground-truth (oracle) perception for the ObjectNav benchmark runtime: the simulator's own annotations in place of the detector and the segmenter.

The method is unchanged -- the same :class:`~sparx_agency.core.planning.object_search.policy.RPTSearchPolicy`,
the same scene graph, room classification, RPT* room order, frontier/FALCON
exploration, side-looks and A* -- and only the two perception services are
swapped for objects that read habitat-sim's semantic sensor. Nothing here is a
benchmark result: a run under it is the method's upper bound under ideal
perception, is marked non-publishable, and exists to separate perception error
from spatial-reasoning and navigation error.

Modules
    ``semantic_frame``   instance boxes and masks from one HxW instance-id frame (numpy only)
    ``hm3d_categories``  HM3D-Semantics raw category names -> the method's detector vocabulary
    ``mp3d_categories``  Matterport3D ``.house`` names and mpcat40 classes -> the MP3D vocabulary
    ``source``           the privileged hand-off: per-scene instance labels and the current frame;
                         the dataset's category table (``HM3D_TABLE`` / ``MP3D_TABLE``) is its one seam
    ``perception``       ``GroundTruthDetector`` / ``GroundTruthSegmenter``, the two contracts
"""
