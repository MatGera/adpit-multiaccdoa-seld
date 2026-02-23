"""Multi-object tracker management (BoT-SORT / ByteTrack)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from vision_fusion.config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class Track:
    """A tracked object."""
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    frames_seen: int = 0
    frames_lost: int = 0
    is_active: bool = True


class TrackerManager:
    """Manages multi-object tracking state.

    Tracking is handled by Ultralytics' built-in BoT-SORT/ByteTrack
    integration. This manager handles post-tracking state management
    and track lifecycle.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tracks: dict[str, dict[int, Track]] = {}  # camera_id → {track_id → Track}

    def update_track(
        self,
        camera_id: str,
        track_id: int,
        class_name: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
    ) -> Track:
        """Update or create a track for a camera."""
        if camera_id not in self._tracks:
            self._tracks[camera_id] = {}

        camera_tracks = self._tracks[camera_id]

        if track_id in camera_tracks:
            track = camera_tracks[track_id]
            track.class_name = class_name
            track.confidence = confidence
            track.bbox = bbox
            track.frames_seen += 1
            track.frames_lost = 0
            track.is_active = True
        else:
            track = Track(
                track_id=track_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                frames_seen=1,
            )
            camera_tracks[track_id] = track

        return track

    def get_active_tracks(self, camera_id: str) -> list[Track]:
        """Get all active tracks for a camera."""
        if camera_id not in self._tracks:
            return []
        return [t for t in self._tracks[camera_id].values() if t.is_active]

    def cleanup_lost_tracks(self, camera_id: str) -> int:
        """Remove tracks that have been lost for too long.

        Returns number of tracks removed.
        """
        if camera_id not in self._tracks:
            return 0

        to_remove = []
        for track_id, track in self._tracks[camera_id].items():
            track.frames_lost += 1
            if track.frames_lost > self._settings.track_buffer:
                track.is_active = False
                to_remove.append(track_id)

        for tid in to_remove:
            del self._tracks[camera_id][tid]

        return len(to_remove)
