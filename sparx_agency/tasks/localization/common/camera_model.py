import numpy as np
import open3d as o3d


def make_intrinsic_from_image(
    width: int,
    height: int,
    hfov_deg: float = 90.0,
) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Create a PinholeCameraIntrinsic object based on image properties.

    This function computes the intrinsic camera parameters for a pinhole camera
    model using the specified image dimensions and horizontal field of view (HFOV).
    The resulting intrinsic parameters are then used to create and return an
    o3d.camera.PinholeCameraIntrinsic object. The focal lengths are derived from
    the HFOV, and the principal point is assumed to be at the image center.

    Parameters:
    width : int
        The width of the image in pixels.
    height : int
        The height of the image in pixels.
    hfov_deg : float, optional
        The horizontal field of view in degrees. Defaults to 90.0.

    Returns:
    o3d.camera.PinholeCameraIntrinsic
        A PinholeCameraIntrinsic object with computed parameters for the given
        image dimensions and field of view.
    """
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    return intrinsic
