"""
Session state management for the prosodic pipeline.

InMemorySessionStore for MVP (single process).
RedisSessionStore for horizontal scaling (serialize SSM/GRU state to Redis).
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionState:
    """
    Complete state for a streaming conversation session.

    Contains everything needed to resume processing if a connection
    drops and reconnects, or to migrate a session between workers.
    """

    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # SSM recurrent state (per-block hidden states)
    # Dict of block_index -> {'ssm_h': tensor, 'conv_buf': tensor}
    ssm_state: Optional[dict] = None

    # ConversationPredictor GRU hidden state
    predictor_state: Optional[Any] = None

    # Rolling prosody history for ConversationPredictor input
    # List of (emotion_probs, vad, confidence) per utterance
    prosody_history: list = field(default_factory=list)

    # Latest outcome predictions
    last_predictions: Optional[dict] = None

    # Frame counter
    frames_processed: int = 0

    # Metadata
    source: str = "unknown"
    vertical: Optional[str] = None
    api_key: Optional[str] = None


class SessionStore(ABC):
    """
    Abstract session state store.

    InMemory for MVP, Redis for scale. The ProsodicPipeline reads/writes
    state through this interface, making it stateless itself.
    """

    @abstractmethod
    async def get(self, session_id: str) -> Optional[SessionState]:
        """Get session state. Returns None if not found."""
        ...

    @abstractmethod
    async def set(self, state: SessionState) -> None:
        """Save session state."""
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete session state."""
        ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        ...

    @abstractmethod
    async def active_count(self) -> int:
        """Number of active sessions."""
        ...


class InMemorySessionStore(SessionStore):
    """
    In-memory session store for single-process deployment.

    Fast, no serialization overhead. Sessions lost on process restart.
    """

    def __init__(self, max_sessions: int = 10000):
        self._sessions: dict[str, SessionState] = {}
        self._max_sessions = max_sessions

    async def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    async def set(self, state: SessionState) -> None:
        state.updated_at = time.time()

        # Evict oldest if at capacity
        if len(self._sessions) >= self._max_sessions and state.session_id not in self._sessions:
            oldest_id = min(self._sessions, key=lambda k: self._sessions[k].updated_at)
            del self._sessions[oldest_id]

        self._sessions[state.session_id] = state

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def active_count(self) -> int:
        return len(self._sessions)

    async def cleanup_stale(self, max_age_seconds: float = 3600) -> int:
        """Remove sessions older than max_age_seconds. Returns count removed."""
        cutoff = time.time() - max_age_seconds
        stale = [sid for sid, s in self._sessions.items() if s.updated_at < cutoff]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)
