
def pointcloud2_to_numpy(msg):
    # read_points returns a structured array with fields 'x', 'y', 'z'
    # Use the helper from sensor_msgs_py
    from sensor_msgs_py import point_cloud2

    # Get the structured array
    points_struct = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

    # Senior trick: Use 'view' to convert structured (x,y,z) to a 2D N x 3 matrix without copying data
    # This is MUCH faster than list(pc_data)
    points_np = np.fromiter(points_struct, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    return np.stack([points_np['x'], points_np['y'], points_np['z']], axis=1)