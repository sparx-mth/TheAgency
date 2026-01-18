#!/usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading

# Import the clean adapters
from sparx_agency.robots.ROBOTICAN.adapters.rooster_control_adapter import PathRunnerNode, parse_path_file
from sparx_agency.robots.ROBOTICAN.adapters.rooster_video_adapter import VideoStreamManager


def main():
    rclpy.init()

    # Configuration
    ROOSTER_ID = "R2"
    # Points to your existing path file in the ROBOTICAN tree
    PATH_FILE = "robots/ROBOTICAN/helpers/txt/roll_custom_path.txt"

    # 1. Initialize Video Adapter
    video_node = VideoStreamManager(
        drone_id=ROOSTER_ID,
        high_resolution=640,
        port=5001
    )

    # 2. Initialize Control Adapter
    flight_node = PathRunnerNode(
        rooster_id=ROOSTER_ID,
        flight_mode=1,  # Requested flight mode in KeepAlive (e.g. 1=GROUND_ROLL, 2=MANUAL)
        arm_before_path=True
    )

    # 3. Setup Executor
    # Using MultiThreadedExecutor is critical so GStreamer and Comms run in parallel
    executor = MultiThreadedExecutor()
    executor.add_node(video_node)
    executor.add_node(flight_node)

    # Start the executor in a background thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        flight_node.get_logger().info(f"--- Launching Demo No 2: VLM on Rooster ({ROOSTER_ID}) ---")

        # Load the flight segments
        segments = parse_path_file(PATH_FILE)

        # Execute flight path (this also triggers the start_capture service)
        flight_node.run_path(segments)

    except KeyboardInterrupt:
        flight_node.get_logger().info("Demo stopped by user.")
    finally:
        flight_node.destroy_node()
        video_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()