import os
import subprocess
import sys
from typing import List

# --- Configuration ---
# ROS 2 default system topics (always present if the daemon is running)
SYSTEM_TOPICS = ['/parameter_events', '/rosout']

# Domain ID range to scan (0 is the default, 101 is the maximum standard ID)
START_ID = 0
END_ID = 101
# ---------------------

def get_topics_for_domain(domain_id: int) -> List[str]:
    """
    Executes 'ros2 topic list' for a specific ROS_DOMAIN_ID and returns the list of topics.
    """
    print(f"-> Scanning Domain ID {domain_id}...", end="", flush=True)

    # Copy current environment variables and set ROS_DOMAIN_ID
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = str(domain_id)

    # Use a timeout to prevent the script from hanging on inactive/slow domains
    try:
        # Run the ros2 command, capturing the output
        result = subprocess.run(
            ['ros2', 'topic', 'list'],
            capture_output=True,
            text=True,
            env=env,
            timeout=5  # Adjust timeout if your network is slow or latency is high
        )

        # Check for errors in the command execution (e.g., ros2 not found)
        if result.returncode != 0:
            print(f" [ERROR: Command failed with code {result.returncode}]")
            # If the command failed for an environment reason, return empty list
            return []

        # Split the output into lines and filter out empty strings
        topics = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return topics

    except FileNotFoundError:
        # Handle case where 'ros2' command is not available in the PATH
        print("\n[CRITICAL ERROR] 'ros2' command not found. Ensure your ROS 2 environment is sourced.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(" [TIMEOUT]")
        return []
    except Exception as e:
        print(f" [UNEXPECTED ERROR: {e}]")
        return []


def main():
    """
    Main function to iterate through domain IDs and analyze topic counts.
    """
    active_domains_found = False
    print(f"--- ROS 2 Domain Scanner (IDs {START_ID} to {END_ID}) ---")

    for domain_id in range(START_ID, END_ID + 1):
        topics = get_topics_for_domain(domain_id)

        # Count the total number of topics found
        topic_count = len(topics)
        
        # Calculate the number of non-system (user-defined) topics
        user_topics = [t for t in topics if t not in SYSTEM_TOPICS]
        user_topic_count = len(user_topics)

        # The condition is: Are there MORE THAN 2 topics?
        # Since system topics are 2, this is equivalent to user_topic_count > 0
        if topic_count > len(SYSTEM_TOPICS):
            active_domains_found = True
            print(f" [ACTIVE] -> Found {topic_count} topics ({user_topic_count} user topic(s)).")
            print(f"   Topics: {topics}")
        elif topic_count == len(SYSTEM_TOPICS):
            print(" [IDLE]   -> Found 2 system topics.")
        elif topic_count > 0:
            # Case where rosout/parameter_events may not have resolved yet, but something is there
            print(f" [LOW]    -> Found {topic_count} topics (less than 2 system topics).")
            print(f"   Topics: {topics}")
        else:
            # Zero topics usually means the domain is completely inactive or unreachable
            print(" [INACTIVE] -> Found 0 topics.")

    print("--------------------------------------------------")
    if not active_domains_found:
        print("Scan complete. No active domains with more than 2 topics found.")
    else:
        print("Scan complete. See ACTIVE domains above.")


if __name__ == "__main__":
    main()
