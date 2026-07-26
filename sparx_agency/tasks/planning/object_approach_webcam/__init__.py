"""Home/laptop test rig for the object-approach target-lock mission.

Runs the exact `tasks/planning/object_approach_offline` stack (detect -> confirm ->
track -> visual servo -> SEARCH/APPROACH/HOVER_LOCK/RECOVER -> HUD) off a laptop
webcam instead of the drone, so the detector/tracker/lock-mode/HUD-colour/RECOVER
mechanisms can be exercised with no drone, no depth, no localization, and no
TensorRT engines. See ``README.md``.
"""
