"""
ProsodicPipeline: continuous prosodic feedback loop.

Ties together:
    FrameExtractor -> ProsodySSM (streaming step) -> ConversationPredictor -> AgentDirective

This is the core product: a continuous prosodic signal that predicts
conversation outcomes and generates real-time adaptation directives
for voice agents (TTS tone, LLM context, intervention triggers).

The pipeline is stateless -- all state lives in SessionStore.
"""

from dataclasses import dataclass, field
from typing import Optional, AsyncIterator, Callable
import asyncio
import time
import logging

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from prosody_ssm.streaming.frame_extractor import FrameExtractor, ProsodyFrame
from prosody_ssm.streaming.bus import AudioBus, AudioFrame
from prosody_ssm.streaming.session import SessionStore, SessionState
from prosody_ssm.model import EmotionLabel

logger = logging.getLogger(__name__)


# --- AgentDirective: the output of the feedback loop ---

@dataclass
class AgentDirective:
    """
    Real-time instruction to the voice agent.
    
    This is what the pipeline produces every N frames. The voice agent
    consumes these to adapt its behavior in real-time.
    """
    
    # TTS adaptation
    tts_emotion: str = "neutral"       # Emotion tag for TTS (Orpheus: happy, sad, neutral, etc.)
    tts_speed: float = 1.0             # Speed multiplier (0.8 = slower, 1.2 = faster)
    
    # LLM context injection
    llm_context: str = ""              # Injected into system prompt for emotional awareness
    
    # Outcome predictions (forward-looking)
    escalation_prob: float = 0.0       # P(conversation escalates), 0-1
    churn_risk: float = 0.0            # P(customer churns within 30 days), 0-1
    resolution_prob: float = 0.5       # P(first-call resolution), 0-1
    predicted_csat: float = 3.0        # Predicted final CSAT, 1-5
    sentiment_forecast: float = 0.0    # Predicted final valence, -1 to +1
    
    # Intervention signals
    intervention_type: str = "none"    # "none", "tone_shift", "escalate_to_human", "offer_callback"
    intervention_urgency: float = 0.0  # 0 = no rush, 1 = immediate
    
    # Current prosodic state (continuous signal, not just a label)
    current_emotion: str = "neutral"   # Most likely emotion (informational)
    emotion_confidence: float = 0.0
    valence: float = 0.0              # Current smoothed valence
    arousal: float = 0.5              # Current smoothed arousal
    
    # Metadata
    confidence: float = 0.0           # Overall directive confidence
    timestamp_ms: float = 0.0
    frames_processed: int = 0


@dataclass
class PipelineOutput:
    """Full output from a pipeline processing cycle."""
    directive: AgentDirective
    prosody_frame: Optional[ProsodyFrame] = None
    emotion_probs: Optional[dict[str, float]] = None
    raw_valence: float = 0.0
    raw_arousal: float = 0.0


# TTS emotion/speed mapping (from conversation.py, centralized here)
_TTS_EMOTION_MAP = {
    "empathetic": "sad",
    "calm": "neutral",
    "enthusiastic": "happy",
    "professional": "neutral",
    "reassuring": "neutral",
    "apologetic": "sad",
}

_TTS_SPEED_MAP = {
    "empathetic": 0.9,
    "calm": 0.85,
    "enthusiastic": 1.1,
    "professional": 1.0,
    "reassuring": 0.95,
    "apologetic": 0.9,
}


def _determine_tone(valence: float, arousal: float, emotion: str) -> str:
    """Determine agent tone from prosodic state."""
    if valence < -0.5 and arousal > 0.6:
        return "calm"
    if emotion == "angry":
        return "calm"
    if emotion == "sad":
        return "empathetic"
    if emotion == "fearful":
        return "reassuring"
    if emotion == "happy":
        return "enthusiastic"
    if valence < -0.3:
        return "empathetic"
    return "professional"


def _determine_intervention(
    escalation_prob: float,
    consecutive_negative: int,
    valence: float,
) -> tuple[str, float]:
    """Determine intervention type and urgency."""
    if escalation_prob > 0.8 and consecutive_negative >= 5:
        return "escalate_to_human", 1.0
    if escalation_prob > 0.6 and consecutive_negative >= 3:
        return "offer_callback", 0.7
    if escalation_prob > 0.4 or valence < -0.5:
        return "tone_shift", 0.4
    return "none", 0.0


class ProsodicPipeline:
    """
    Continuous prosodic feedback loop.
    
    Processes audio frames from the AudioBus, maintains SSM state
    via SessionStore, runs outcome prediction, and produces
    AgentDirectives for voice agent adaptation.
    
    The pipeline itself is stateless -- all persistent state lives
    in the SessionStore, making it safe for horizontal scaling.
    
    Usage:
        pipeline = ProsodicPipeline(model, bus, store)
        
        # Start processing a session
        async for output in pipeline.run(session_id):
            # Send directive to voice agent
            agent.adapt(output.directive)
    """
    
    # How often to emit a directive (every N frames)
    DIRECTIVE_INTERVAL = 4  # Every 4 frames = every 200ms at 50ms frames
    
    def __init__(
        self,
        model=None,  # ProsodySSMClassifier (optional -- heuristic mode if None)
        predictor=None,  # ConversationPredictor (optional)
        bus: Optional[AudioBus] = None,
        store: Optional[SessionStore] = None,
        sample_rate: int = 16000,
        directive_interval: int = 4,
    ):
        self.model = model
        self.predictor = predictor
        self.bus = bus
        self.store = store
        self.sample_rate = sample_rate
        self.DIRECTIVE_INTERVAL = directive_interval
        
        # Frame extractor is per-pipeline-instance (lightweight)
        self._extractors: dict[str, FrameExtractor] = {}
    
    def _get_extractor(self, session_id: str) -> FrameExtractor:
        if session_id not in self._extractors:
            self._extractors[session_id] = FrameExtractor(sample_rate=self.sample_rate)
        return self._extractors[session_id]
    
    async def process_frame(
        self,
        session_id: str,
        pcm_data: bytes,
    ) -> Optional[PipelineOutput]:
        """
        Process a single audio chunk for a session.
        
        Call this directly when not using the AudioBus (e.g., from a
        WebSocket handler). Returns a PipelineOutput when a directive
        is ready (every DIRECTIVE_INTERVAL frames), None otherwise.
        """
        extractor = self._get_extractor(session_id)
        
        # Extract prosody frames from audio
        frames = extractor.process_frames(pcm_data)
        if not frames:
            return None
        
        # Get or create session state
        state = None
        if self.store:
            state = await self.store.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
        
        # Process each prosody frame through SSM
        last_output = None
        for frame in frames:
            state.frames_processed += 1
            last_output = self._process_prosody_frame(frame, state)
        
        # Save state
        if self.store:
            await self.store.set(state)
        
        # Emit directive at interval
        if state.frames_processed % self.DIRECTIVE_INTERVAL == 0 and last_output:
            return last_output
        
        return None
    
    def _process_prosody_frame(
        self,
        frame: ProsodyFrame,
        state: SessionState,
    ) -> PipelineOutput:
        """Process a single prosody frame through SSM and prediction."""
        
        emotion_probs_dict = {}
        valence = 0.0
        arousal = 0.5
        dominance = 0.5
        current_emotion = "neutral"
        confidence = 0.0
        
        if TORCH_AVAILABLE and self.model is not None:
            # Run SSM step with persistent state
            prosody_vec = torch.from_numpy(frame.to_vector()).float().unsqueeze(0)
            device = next(self.model.parameters()).device
            prosody_vec = prosody_vec.to(device)
            
            probs, vad, new_ssm_state = self.model.step(
                prosody_vec,
                state=state.ssm_state,
            )
            state.ssm_state = new_ssm_state
            
            # Extract results
            probs_np = probs[0].cpu().numpy()
            vad_np = vad[0].cpu().numpy()
            
            emotions = list(EmotionLabel)
            emotion_probs_dict = {e.value: float(probs_np[i]) for i, e in enumerate(emotions) if i < len(probs_np)}
            
            best_idx = int(np.argmax(probs_np))
            current_emotion = emotions[best_idx].value if best_idx < len(emotions) else "neutral"
            confidence = float(probs_np[best_idx])
            
            valence = float(vad_np[0]) if len(vad_np) > 0 else 0.0
            arousal = float(vad_np[1]) if len(vad_np) > 1 else 0.5
            dominance = float(vad_np[2]) if len(vad_np) > 2 else 0.5
        else:
            # Heuristic mode: derive emotion from prosody features directly
            if frame.energy > 0.1 and frame.f0_mean > 200:
                current_emotion = "angry"
                valence = -0.5
                arousal = 0.8
            elif frame.energy < 0.02:
                current_emotion = "sad"
                valence = -0.3
                arousal = 0.3
            else:
                current_emotion = "neutral"
                valence = 0.0
                arousal = 0.5
            confidence = 0.5
            emotion_probs_dict = {current_emotion: confidence}
        
        # Track history for ConversationPredictor
        state.prosody_history.append({
            "emotion_probs": list(emotion_probs_dict.values()),
            "vad": [valence, arousal, dominance],
            "confidence": confidence,
            "timestamp": frame.timestamp_ms,
        })
        # Keep rolling window
        if len(state.prosody_history) > 50:
            state.prosody_history = state.prosody_history[-50:]
        
        # Run ConversationPredictor if available
        escalation_prob = 0.0
        churn_risk = 0.0
        resolution_prob = 0.5
        predicted_csat = 3.0
        sentiment_forecast = 0.0
        recommended_tone = "professional"
        
        if TORCH_AVAILABLE and self.predictor is not None and len(state.prosody_history) >= 3:
            try:
                from prosody_ssm.conversation_model import ForwardPrediction
                pred = self.predictor.predict(
                    [h["emotion_probs"] for h in state.prosody_history],
                    [h["vad"] for h in state.prosody_history],
                    [h["confidence"] for h in state.prosody_history],
                )
                if pred is not None:
                    escalation_prob = pred.will_escalate
                    churn_risk = pred.churn_risk
                    resolution_prob = pred.resolution_prob
                    predicted_csat = pred.final_csat
                    sentiment_forecast = pred.sentiment_forecast
                    recommended_tone = pred.recommended_tone.value
            except Exception as e:
                logger.debug(f"ConversationPredictor error: {e}")
        
        # Determine tone and intervention from prosodic state
        if recommended_tone == "professional":
            recommended_tone = _determine_tone(valence, arousal, current_emotion)
        
        # Count consecutive negative
        consecutive_neg = 0
        for h in reversed(state.prosody_history):
            if h["vad"][0] < -0.3:
                consecutive_neg += 1
            else:
                break
        
        intervention_type, intervention_urgency = _determine_intervention(
            escalation_prob, consecutive_neg, valence
        )
        
        # Build directive
        tts_emotion = _TTS_EMOTION_MAP.get(recommended_tone, "neutral")
        tts_speed = _TTS_SPEED_MAP.get(recommended_tone, 1.0)
        
        # LLM context
        llm_context = (
            f"Customer emotion: {current_emotion} (valence={valence:+.2f}, "
            f"arousal={arousal:.2f}). Escalation risk: {escalation_prob:.0%}. "
            f"Use {recommended_tone} tone."
        )
        if intervention_type != "none":
            llm_context += f" Intervention: {intervention_type}."
        
        directive = AgentDirective(
            tts_emotion=tts_emotion,
            tts_speed=tts_speed,
            llm_context=llm_context,
            escalation_prob=escalation_prob,
            churn_risk=churn_risk,
            resolution_prob=resolution_prob,
            predicted_csat=predicted_csat,
            sentiment_forecast=sentiment_forecast,
            intervention_type=intervention_type,
            intervention_urgency=intervention_urgency,
            current_emotion=current_emotion,
            emotion_confidence=confidence,
            valence=valence,
            arousal=arousal,
            confidence=confidence,
            timestamp_ms=frame.timestamp_ms,
            frames_processed=state.frames_processed,
        )
        
        return PipelineOutput(
            directive=directive,
            prosody_frame=frame,
            emotion_probs=emotion_probs_dict,
            raw_valence=valence,
            raw_arousal=arousal,
        )
    
    async def run(self, session_id: str) -> AsyncIterator[PipelineOutput]:
        """
        Run the pipeline for a session, consuming from the AudioBus.
        
        Yields PipelineOutput (including AgentDirective) at regular intervals.
        Runs until the session is closed on the bus.
        """
        if self.bus is None:
            raise RuntimeError("AudioBus required for run(). Use process_frame() for direct calls.")
        
        async for audio_frame in self.bus.subscribe(session_id):
            output = await self.process_frame(session_id, audio_frame.pcm_data)
            if output is not None:
                yield output
    
    async def close_session(self, session_id: str) -> None:
        """Clean up session resources."""
        self._extractors.pop(session_id, None)
        if self.store:
            await self.store.delete(session_id)
        if self.bus:
            await self.bus.close_session(session_id)
