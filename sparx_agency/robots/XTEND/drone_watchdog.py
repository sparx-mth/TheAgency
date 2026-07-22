import asyncio
import psutil
import sys
from sparx_agency.robots.XTEND.automation import ControllerAutomation

# --- CONFIGURATION ---
# Match these to your actual drone/network setup
DRONE_HOST = "192.0.0.15"
DRONE_PORT = 8000
ROBOT_UID = "drndfb3eeb1"
TARGET_SCRIPT = "xtend_dome_main.py"


# ---------------------

class SafetyWatchdog(ControllerAutomation):
    def __init__(self, host, port, robot_uid):
        # We set a lower frequency for the monitor to save bandwidth,
        # but you can keep it at 30.0 if preferred.
        super().__init__(host, port, 30.0, robot_uid)
        self.is_armed = False
        self.main_script_running = True

    def check_process(self):
        """Checks if the navigation script is in the system process list."""
        for proc in psutil.process_iter(['cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and any(TARGET_SCRIPT in arg for arg in cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    async def monitor_loop(self):
        """The brain of the watchdog."""
        print(f"🛡️ Guarding: Looking for {TARGET_SCRIPT}...")

        while True:
            self.main_script_running = self.check_process()

            # LOGIC: If main script is DEAD but we think the drone is ARMED
            # Note: You can also just force land if script_running is False
            # to be extra safe, regardless of telemetry.
            if not self.main_script_running:
                print(f"⚠️ ALERT: {TARGET_SCRIPT} not found! Initiating Emergency Recovery...")

                # Execute your automation class methods
                await self.move_down(500)
                await asyncio.sleep(2)
                await self.move_down(300)
                await asyncio.sleep(2)
                self.main_script_running = False
                await self.land()
                await asyncio.sleep(2)
                await self.disarm_robot()

                print("Recovery commands sent. Waiting for script to restart...")
                # Sleep longer after recovery to avoid spamming
                await asyncio.sleep(5)

            await asyncio.sleep(1)

    async def run_watchdog(self):
        """Starts the standard communication plus our monitor."""
        # Standard send/receive loops from your automation.py
        await asyncio.gather(
            self.run_communication(),  # This starts your send_message/receive_message
            self.monitor_loop()
        )

    async def run_communication(self):
        """
        Modified version of your run() to keep comms open
        without running a static scenario.
        """
        import websockets
        async with websockets.connect(self.uri) as websocket:
            send_task = asyncio.create_task(self.send_message(websocket))
            receive_task = asyncio.create_task(self.receive_message(websocket))
            await asyncio.gather(send_task, receive_task)


if __name__ == "__main__":
    guard = SafetyWatchdog(DRONE_HOST, DRONE_PORT, ROBOT_UID)
    try:
        asyncio.run(guard.run_watchdog())
    except KeyboardInterrupt:
        print("\nWatchdog deactivated.")