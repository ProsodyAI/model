"""
Streaming prosodic pipeline for real-time conversation analysis.

Provides frame-level prosody extraction, SSM state management,
outcome prediction, and agent adaptation directives.
"""

from prosody_ssm.streaming.bus import AudioBus, AudioFrame, WebSocketAudioBus
from prosody_ssm.streaming.frame_extractor import FrameExtractor, ProsodyFrame
from prosody_ssm.streaming.pipeline import AgentDirective, PipelineOutput, ProsodicPipeline
from prosody_ssm.streaming.session import InMemorySessionStore, SessionState, SessionStore

__all__ = [
    "FrameExtractor",
    "ProsodyFrame",
    "AudioBus",
    "AudioFrame",
    "WebSocketAudioBus",
    "SessionStore",
    "InMemorySessionStore",
    "SessionState",
    "ProsodicPipeline",
    "AgentDirective",
    "PipelineOutput",
]
