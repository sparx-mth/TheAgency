"""Optional recorder wrappers around the shared runner, never a second run loop."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2

from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError
from sparx_agency.tasks.planning.objnav_benchmark.results_io import strict_json, write_atomically
from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import (
	FRAME_SIZE, method_snapshot, render_dashboard)


def ffmpeg_executable():
	"""Use installed FFmpeg or imageio's bundled executable, with no download."""
	candidates = [os.environ.get("IMAGEIO_FFMPEG_EXE"), shutil.which("ffmpeg"),
				  str(Path(sys.executable).parent / "ffmpeg")]
	try:
		import imageio_ffmpeg
		candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
	except (ImportError, RuntimeError):
		pass
	for candidate in dict.fromkeys(candidates):
		if not candidate:
			continue
		try:
			result = subprocess.run([candidate, "-version"], stdout=subprocess.DEVNULL,
									stderr=subprocess.DEVNULL, timeout=5)
			if result.returncode == 0:
				return candidate
		except (OSError, subprocess.TimeoutExpired):
			continue
	raise HarnessError("No working FFmpeg: set IMAGEIO_FFMPEG_EXE to an installed encoder")


class VideoSink:
	"""Streaming, browser-playable H.264; memory does not grow with episode length."""

	def __init__(self, path, fps):
		self.path = Path(path)
		self.temporary = self.path.with_name("video.partial.mp4")
		self.log = self.path.with_suffix(".encoder.log").open("wb")
		self.process = subprocess.Popen([
			ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
			"-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % FRAME_SIZE,
			"-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
			"-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.temporary)],
			stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)

	def write(self, frame):
		try:
			self.process.stdin.write(frame.tobytes())
		except (BrokenPipeError, OSError) as exc:
			raise HarnessError("Video encoder stopped; see its .encoder.log") from exc

	def close(self):
		try:
			self.process.stdin.close()
			if self.process.wait(timeout=60) != 0:
				raise HarnessError("Video encoding failed; see %s" % self.log.name)
			self.temporary.replace(self.path)
		finally:
			if self.process.poll() is None:
				self.process.kill()
				self.process.wait()
			self.log.close()


class PolicyProbe:
	"""Observe the exact command that reaches the converter, including approach paths."""

	def __init__(self, policy):
		self.policy, self.command = policy, None
		self.name = policy.name

	def reset(self, episode, target):
		self.command = None
		return self.policy.reset(episode, target)

	def plan(self, observation):
		self.command = self.policy.plan(observation)
		return self.command

	def notify_blocked(self, observation):
		callback = getattr(self.policy, "notify_blocked", None)
		if callback:
			callback(observation)

	def notify_action(self, observation, action):
		callback = getattr(self.policy, "notify_action", None)
		if callback:
			callback(observation, action)

	def episode_info(self):
		callback = getattr(self.policy, "episode_info", None)
		return callback() if callback is not None else {}


class EpisodeRecorder:
	"""Record actual observations/decisions and evaluator-only telemetry separately."""

	def __init__(self, root, policy, fps=6, writer_factory=VideoSink, selected_episode_ids=None):
		self.root, self.policy = Path(root), policy
		self.fps, self.writer_factory = fps, writer_factory
		self.selected_episode_ids = None if selected_episode_ids is None else set(selected_episode_ids)
		self.enabled = True
		self.video = self.trace = self.csv_file = None
		self.episode_dir = None
		self.last_frame = None

	def begin(self, episode, observation, telemetry):
		self.close()
		self.episode_dir = None
		self.last_frame = None
		self.enabled = self.selected_episode_ids is None or episode.episode_id in self.selected_episode_ids
		if not self.enabled:
			return
		self.episode = episode
		self.started = time.monotonic()
		self.last_snapshot = None
		self.trail = [(observation.pose.x, observation.pose.y)]
		self.floor_trails = {}
		# Qualified ids are hashed into directories to avoid path traversal.
		import hashlib
		key = hashlib.sha256(episode.episode_id.encode()).hexdigest()[:12]
		self.episode_dir = self.root / "recordings" / key
		self.episode_dir.mkdir(parents=True, exist_ok=True)
		self.trace = (self.episode_dir / "steps.jsonl").open("w", encoding="utf-8")
		self.csv_file = (self.episode_dir / "trajectory.csv").open("w", newline="", encoding="utf-8")
		self.csv = csv.DictWriter(self.csv_file, fieldnames=(
			"step", "x", "y", "z", "yaw", "camera_pitch", "distance_to_goal_m", "path_length_m", "wall_s"))
		self.csv.writeheader()
		write_atomically(self.episode_dir / "episode.json", strict_json({
			"episode_id": episode.episode_id, "scene": episode.scene_id,
			"target": episode.target_category, "video_fps": self.fps,
			"video_timebase": "one frame per decision; accelerated, not real time"}, "recording", indent=2))
		self._position(observation, telemetry)

	def _position(self, observation, telemetry):
		pose = observation.pose
		row = dict(step=int(observation.step), x=pose.x, y=pose.y, z=pose.z,
				   yaw=pose.yaw, camera_pitch=pose.camera_pitch,
				   distance_to_goal_m=telemetry.get("distance_to_goal_m"),
				   path_length_m=telemetry.get("path_length_m"), wall_s=time.monotonic() - self.started)
		self.csv.writerow(row)
		self.csv_file.flush()

	def decision(self, observation, decision, command):
		if not self.enabled:
			return
		try:
			snapshot = method_snapshot(self.policy)
			if command is not None:
				snapshot["planned_path"] = [list(p) for p in command.waypoints]
			self.last_snapshot = snapshot
			detail = {"action": decision.action.name, "info": dict(decision.info)}
			row = {"step": int(observation.step), "pose": asdict(observation.pose),
				   "decision": detail, "method": snapshot}
			self.trace.write(strict_json(row, "step recording") + "\n")
			self.trace.flush()
			self._frame(observation, detail, snapshot)
		except HarnessError:
			raise
		except Exception as exc:
			raise HarnessError("Recording failed, not an agent failure: %s" % exc) from exc

	def after_step(self, observation, telemetry, terminal):
		if not self.enabled:
			return
		self.trail.append((observation.pose.x, observation.pose.y))
		self._position(observation, telemetry)
		if terminal:
			snapshot = self.last_snapshot or method_snapshot(self.policy)
			self._frame(observation, {}, snapshot, final=True)

	def _frame(self, observation, decision, snapshot, final=False):
		floor_id = snapshot.get("floor_id", 0)
		trail = self.floor_trails.setdefault(floor_id, [])
		mapping = getattr(self.policy, "mapping", None)
		if not getattr(getattr(mapping, "atlas", None), "in_transition", False):
			point = (observation.pose.x, observation.pose.y)
			if not trail or trail[-1] != point:
				trail.append(point)
		frame = render_dashboard(self.policy, observation, trail, decision,
								 self.episode.episode_id, snapshot, final=final)
		if self.video is None:
			self.video = self.writer_factory(self.episode_dir / "video.mp4", self.fps)
		self.video.write(frame)
		self.last_frame = frame
		temporary = self.root / "latest.tmp.jpg"
		if not cv2.imwrite(str(temporary), frame):
			raise HarnessError("Cannot write live preview")
		temporary.replace(self.root / "latest.jpg")
		write_atomically(self.root / "live.json", strict_json({
			"episode_id": self.episode.episode_id, "step": int(observation.step),
			"action": decision.get("action", "terminal"), "state": snapshot["state"],
			"objects": len(snapshot["objects"]), "completed": False}, "live status"))

	def complete(self, record):
		if self.episode_dir is not None:
			write_atomically(self.episode_dir / "metrics.json", strict_json(record.to_row(), "episode metrics", indent=2))
			if self.last_frame is not None:
				cv2.imwrite(str(self.episode_dir / "final.jpg"), self.last_frame)
		self.close()

	def close(self):
		video, self.video = self.video, None
		try:
			if video is not None:
				video.close()
		finally:
			for name in ("trace", "csv_file"):
				stream = getattr(self, name)
				if stream is not None:
					stream.close()
					setattr(self, name, None)


class RecordingAgent(ObjNavAgent):
	def __init__(self, agent, probe, recorder):
		self.agent, self.probe, self.recorder = agent, probe, recorder

	@property
	def name(self):
		return self.agent.name

	def reset(self, episode):
		self.agent.reset(episode)

	def act(self, observation):
		decision = self.agent.act(observation)
		self.recorder.decision(observation, decision, self.probe.command)
		return decision

	def episode_info(self):
		return self.agent.episode_info()


class RecordingEnv(ObjNavEnv):
	def __init__(self, env, recorder):
		self.env, self.recorder = env, recorder

	@property
	def name(self):
		return self.env.name

	def episode_ids(self):
		return self.env.episode_ids()

	def reset(self, episode_id):
		episode, observation = self.env.reset(episode_id)
		self.recorder.begin(episode, observation, self.env.evaluation_diagnostics())
		return episode, observation

	def step(self, action):
		observation = self.env.step(action)
		self.recorder.after_step(observation, self.env.evaluation_diagnostics(), self.env.episode_over)
		return observation

	@property
	def episode_over(self):
		return self.env.episode_over

	def measure(self):
		return self.env.measure()

	def close(self):
		try:
			self.recorder.close()
		finally:
			self.env.close()




