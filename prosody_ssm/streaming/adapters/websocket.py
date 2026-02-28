"""
WebSocket ingress adapter.

Converts raw WebSocket PCM frames into AudioFrame objects
and publishes them to the AudioBus.
"""

import base64
import json
import logging
from typing import Optional

from prosody_ssm.streaming.bus import AudioBus, AudioFrame

logger = logging.getLogger(__name__)


class WebSocketAdapter:
    """
    Adapter that bridges a WebSocket connection to the AudioBus.
    
    Handles:
    - Binary messages as raw PCM int16
    - JSON messages with base64-encoded audio
    - Session lifecycle (create, close)
    """
    
    def __init__(self, bus: AudioBus, sample_rate: int = 16000):
        self.bus = bus
        self.sample_rate = sample_rate
        self._sequence: dict[str, int] = {}
    
    async def handle_message(
        self,
        session_id: str,
        data: bytes | str,
        source: str = "websocket",
    ) -> None:
        """
        Handle an incoming WebSocket message.
        
        Args:
            session_id: Session identifier
            data: Raw bytes (PCM) or JSON string with base64 audio
            source: Source identifier for metadata
        """
        pcm_data: Optional[bytes] = None
        
        if isinstance(data, bytes):
            # Raw PCM int16
            pcm_data = data
        elif isinstance(data, str):
            # JSON with base64 audio
            try:
                msg = json.loads(data)
                if "audio" in msg:
                    pcm_data = base64.b64decode(msg["audio"])
                elif "pcm" in msg:
                    pcm_data = base64.b64decode(msg["pcm"])
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Invalid JSON message for session {session_id}")
                return
        
        if pcm_data is None or len(pcm_data) == 0:
            return
        
        seq = self._sequence.get(session_id, 0)
        self._sequence[session_id] = seq + 1
        
        frame = AudioFrame(
            session_id=session_id,
            pcm_data=pcm_data,
            sample_rate=self.sample_rate,
            sequence=seq,
            source=source,
        )
        
        await self.bus.publish(frame)
    
    async def close_session(self, session_id: str) -> None:
        """Signal end of session."""
        self._sequence.pop(session_id, None)
        await self.bus.close_session(session_id)
