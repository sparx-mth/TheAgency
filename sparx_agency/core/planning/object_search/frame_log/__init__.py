"""The frame log: every RGB frame of an episode kept under its exact capture pose, on disk or in RAM.

See ``README.md`` beside this file.
"""
from sparx_agency.core.planning.object_search.frame_log.codec import FrameCodec
from sparx_agency.core.planning.object_search.frame_log.disk_store import (
    DiskFrameStore, episode_directory_name, read_frame_log)
from sparx_agency.core.planning.object_search.frame_log.episode_frame_log import FRAME_LOG_DIR_ENV, EpisodeFrameLog
from sparx_agency.core.planning.object_search.frame_log.interfaces import (
    FrameLogError, FrameRecord, FrameStore, frame_name)
from sparx_agency.core.planning.object_search.frame_log.memory_store import MemoryFrameStore

__all__ = ["DiskFrameStore", "EpisodeFrameLog", "FRAME_LOG_DIR_ENV", "FrameCodec", "FrameLogError", "FrameRecord",
           "FrameStore", "MemoryFrameStore", "episode_directory_name", "frame_name", "read_frame_log"]
