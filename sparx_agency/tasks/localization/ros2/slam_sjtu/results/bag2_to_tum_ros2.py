#!/usr/bin/env python3
import sys
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def export_topic_to_tum(bag_dir: str, topic: str, out_path: str):
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_dir,
        storage_id='sqlite3'
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics_and_types}

    if topic not in type_map:
        available = "\n".join(sorted(type_map.keys()))
        raise SystemExit(f"Topic '{topic}' not found. Available:\n{available}")

    msg_type_str = type_map[topic]
    msg_cls = get_message(msg_type_str)

    n = 0
    with open(out_path, "w") as f:
        while reader.has_next():
            (t_name, raw, t_ns) = reader.read_next()
            if t_name != topic:
                continue

            msg = deserialize_message(raw, msg_cls)

            # bag timestamp in seconds
            t = t_ns * 1e-9

            # supports PoseStamped (msg.pose) and Pose (msg)
            pose = msg.pose if hasattr(msg, "pose") else msg

            p = pose.position
            q = pose.orientation

            f.write(f"{t:.9f} {p.x} {p.y} {p.z} {q.x} {q.y} {q.z} {q.w}\n")
            n += 1

    print(f"Wrote {n} poses from {topic} ({msg_type_str}) -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 bag2_to_tum_ros2.py <bag_dir> <topic> <out_tum.txt>")
        sys.exit(1)

    export_topic_to_tum(sys.argv[1], sys.argv[2], sys.argv[3])
