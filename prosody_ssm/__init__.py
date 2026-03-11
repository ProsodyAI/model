"""
ProsodySSM: Prosody analysis and KPI outcome prediction using State Space Models.

We pretrain ProsodySSM on public emotion datasets to learn prosodic
representations, then adapt to client-defined KPIs via a KPI-conditioned
head when outcome labels are available.

This package provides:
- Prosodic feature extraction (pitch, energy, rhythm, phonemes)
- SSM backbone for temporal prosodic pattern learning
- Emotion classification (pretraining head — learns representations)
- KPI-conditioned outcome prediction (downstream head — requires outcome data)
- Conversation-level tracking with TTS/LLM integration
"""

from prosody_ssm.conversation import (
    AgentTone,
    ConversationEmotionState,
    EmotionTracker,
    EmotionTrajectory,
    EscalationRisk,
    format_emotion_context_compact,
    format_emotion_context_for_llm,
    get_tts_params_for_tone,
)
from prosody_ssm.emotions import Emotion, EmotionAnnotator
from prosody_ssm.features import PhoneticFeatureExtractor, ProsodyFeatureExtractor
from prosody_ssm.model import (
    # Legacy — kept for existing code
    EmotionLabel,
    EmotionPrediction,
    KPIModelConfig,
    KPIModelOutput,
    ProsodyPrediction,
    ProsodySSM,
    ProsodySSMClassifier,  # backward-compat alias
)

__version__ = "0.2.0"
__all__ = [
    # Feature extraction
    "ProsodyFeatureExtractor",
    "PhoneticFeatureExtractor",
    # Model (new)
    "ProsodySSM",
    "ProsodyPrediction",
    "KPIModelConfig",
    "KPIModelOutput",
    # Model (pretraining head)
    "ProsodySSMClassifier",
    "EmotionLabel",
    "EmotionPrediction",
    # Annotation
    "EmotionAnnotator",
    "Emotion",
    # Conversation tracking
    "EmotionTracker",
    "ConversationEmotionState",
    "EmotionTrajectory",
    "EscalationRisk",
    "AgentTone",
    # TTS/LLM integration
    "get_tts_params_for_tone",
    "format_emotion_context_for_llm",
    "format_emotion_context_compact",
]
