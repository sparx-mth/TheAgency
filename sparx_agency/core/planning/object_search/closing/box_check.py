"""The crop, the prompt and the question words of the vision-model check before a STOP (the model call lives in the runtime adapter)."""
import numpy as np

CROP_MARGIN = 0.5
CROP_MIN_SIDE_PX = 224
RED = (255, 0, 0)
# The benchmark's category names as the question words, the wording Daphna approved (2026-11-16).
CATEGORY_PHRASES = {"sofa": "couch", "tv_monitor": "tv", "plant": "potted plant"}


def category_phrase(category):
    return CATEGORY_PHRASES.get(category, category.replace("_", " "))


def box_prompt(phrase):
    return ('In this photo, look only at the object inside the red rectangle. '
            'Is it a %s? Answer with JSON {"answer": "yes"} or {"answer": "no"}.' % phrase)


def crop_around_box(rgb, xyxy, margin=CROP_MARGIN, min_side=CROP_MIN_SIDE_PX):
    """The frame around the box, grown by ``margin`` of its size and at least ``min_side`` px, with the box drawn in red."""
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = xyxy
    centre_x, centre_y = (x0 + x1) / 2, (y0 + y1) / 2
    half_width = max((x1 - x0) * (0.5 + margin), min_side / 2)
    half_height = max((y1 - y0) * (0.5 + margin), min_side / 2)
    left, right = int(max(0, centre_x - half_width)), int(min(width, centre_x + half_width))
    top, bottom = int(max(0, centre_y - half_height)), int(min(height, centre_y + half_height))
    crop = rgb[top:bottom, left:right].copy()
    return _draw_rectangle(crop, (x0 - left, y0 - top, x1 - left, y1 - top))


def _draw_rectangle(image, xyxy, thickness=2):
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in xyxy)
    columns = slice(max(0, x0), min(width, x1 + 1))
    rows = slice(max(0, y0), min(height, y1 + 1))
    for edge in (y0, y1):
        image[max(0, edge):min(height, edge + thickness), columns] = RED
    for edge in (x0, x1):
        image[rows, max(0, edge):min(width, edge + thickness)] = RED
    return image
