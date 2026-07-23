# __init__.py
from .math_utils import clamp, clamp_symmetric, clamp_axis
from .logging_utils import TicToc

__all__ = ['clamp', 'clamp_symmetric', 'clamp_axis', 'TicToc']

# txt_utils/image_utils pull in cv2, which isn't installed in every runtime
# that needs this package (e.g. the ROBOTICAN Rooster container). Guard them
# so lean consumers (math_utils, TicToc) still work without cv2.
try:
    from .txt_utils import (
        strip_leading_slash,
        stamp_to_sec,
        pose_to_name,
        parse_pose_from_name,
        update_sidecar_json
    )
    from .image_utils import (
        ros_image_to_rgb_np,
        numpy_to_image_msg,
        apply_crop_and_flip,
        list_frames, get_objects_via_histogram
    )
    __all__ += [
        'strip_leading_slash', 'stamp_to_sec', 'pose_to_name',
        'parse_pose_from_name', 'update_sidecar_json',
        'ros_image_to_rgb_np', 'numpy_to_image_msg',
        'apply_crop_and_flip', 'list_frames', 'get_objects_via_histogram',
    ]
except ImportError:
    pass