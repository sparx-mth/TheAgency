#!/usr/bin/env python3
"""
run_pipeline_loop.py - Runs pipeline every 15 seconds
"""

import time
import subprocess
from datetime import datetime

INTERVAL_SECONDS = 15


def run_pipeline_once():
    """Run the pipeline script once"""
    try:
        print(f"\n{'=' * 60}")
        print(f"Running pipeline at {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'=' * 60}")

        subprocess.run(["python3", "pipeline.py"], check=True)

        print(f"\n Pipeline completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n Pipeline failed with error: {e}")
        return False
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        return False


def main():
    print("Starting pipeline loop (every 15 seconds)")
    print("Press Ctrl+C to stop\n")

    run_count = 0

    try:
        while True:
            run_count += 1
            print(f"\n--- Run #{run_count} ---")

            run_pipeline_once()

            print(f"\nWaiting {INTERVAL_SECONDS} seconds until next run...")
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print(f"\n\nStopped after {run_count} runs")
        print("Goodbye!")


if __name__ == "__main__":
    main()