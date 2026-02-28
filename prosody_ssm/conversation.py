"""
Conversation-level emotion tracking and adaptation.

Tracks user emotional state over time and provides:
1. Emotion trajectory (improving/stable/declining)
2. TTS adaptation recommendations (what tone should the agent use)
3. LLM context injection (emotional awareness for response generation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING
from collections import deque
import time

from prosody_ssm.model import EmotionLabel, EmotionPrediction
from prosody_ssm.conversation_model import ForwardPrediction, RecommendedTone

if TYPE_CHECKING:
    from prosody_ssm.conversation_model import ConversationPredictor


class EmotionTrajectory(str, Enum):
    """Direction of emotional change over conversation."""
    IMPROVING = "improving"      # Moving toward positive
    STABLE = "stable"            # No significant change
    DECLINING = "declining"      # Moving toward negative
    VOLATILE = "volatile"        # Rapid changes


class EscalationRisk(str, Enum):
    """Risk level for conversation escalation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentTone(str, Enum):
    """Recommended tone for agent TTS response."""
    EMPATHETIC = "empathetic"    # Warm, understanding (for frustrated/sad users)
    CALM = "calm"                # Steady, de-escalating (for angry users)
    ENTHUSIASTIC = "enthusiastic"  # Upbeat, matching (for happy users)
    PROFESSIONAL = "professional"  # Neutral, clear (default)
    REASSURING = "reassuring"    # Confident, supportive (for anxious users)
    APOLOGETIC = "apologetic"    # Sorry, acknowledging (after complaints)


@dataclass
class EmotionSnapshot:
    """Single emotion reading with timestamp."""
    emotion: EmotionLabel
    confidence: float
    valence: float
    arousal: float
    dominance: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationEmotionState:
    """Aggregated emotional state for a conversation."""
    
    # Current state (raw, per-utterance)
    current_emotion: EmotionLabel
    current_confidence: float
    current_valence: float
    current_arousal: float
    current_dominance: float
    
    # Smoothed state (EMA-filtered, for decision-making)
    smoothed_valence: float
    smoothed_arousal: float
    
    # Spike detection
    spike_detected: bool  # True when raw deviates significantly from smoothed
    consecutive_negative: int  # Consecutive readings below negative threshold
    
    # Trajectory
    trajectory: EmotionTrajectory
    valence_trend: float  # Positive = improving, negative = declining
    
    # Risk assessment (based on smoothed values, not raw)
    escalation_risk: EscalationRisk
    negative_emotion_count: int
    peak_negative_valence: float
    
    # Aggregates
    avg_valence: float
    avg_arousal: float
    segment_count: int
    
    # Recommendations
    recommended_tone: AgentTone
    coaching_hint: str
    
    # Forward predictions (from ConversationPredictor, None when predictor unavailable)
    forward_predictions: Optional[ForwardPrediction] = None


class EmotionTracker:
    """
    Tracks emotional state across a conversation.
    
    Maintains a sliding window of emotion readings and computes
    aggregate metrics for trajectory analysis and TTS adaptation.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        trajectory_threshold: float = 0.15,
        escalation_valence_threshold: float = -0.5,
        predictor: Optional[ConversationPredictor] = None,
        smoothing_alpha: float = 0.3,
        escalation_consecutive_threshold: int = 3,
        spike_deviation_threshold: float = 0.4,
    ):
        """
        Initialize tracker.
        
        Args:
            window_size: Number of recent readings to consider
            trajectory_threshold: Valence change to count as improving/declining
            escalation_valence_threshold: Valence below which escalation risk increases
            predictor: Optional ConversationPredictor for forward-looking predictions
            smoothing_alpha: EMA decay factor (0-1). Lower = more smoothing, higher = more reactive.
                0.3 means each new reading contributes 30% to the smoothed value.
            escalation_consecutive_threshold: Number of consecutive negative readings
                required before escalating to HIGH/CRITICAL.
            spike_deviation_threshold: Absolute difference between raw and smoothed
                valence that triggers spike_detected flag.
        """
        self.window_size = window_size
        self.trajectory_threshold = trajectory_threshold
        self.escalation_valence_threshold = escalation_valence_threshold
        self._predictor = predictor
        self.smoothing_alpha = smoothing_alpha
        self.escalation_consecutive_threshold = escalation_consecutive_threshold
        self.spike_deviation_threshold = spike_deviation_threshold
        
        self._history: deque[EmotionSnapshot] = deque(maxlen=window_size)
        self._all_readings: list[EmotionSnapshot] = []
        self._negative_count = 0
        self._peak_negative_valence = 0.0
        
        # EMA smoothed values (initialized on first reading)
        self._smoothed_valence: Optional[float] = None
        self._smoothed_arousal: Optional[float] = None
        
        # Consecutive negative reading counter (resets when valence goes positive)
        self._consecutive_negative = 0
        
        # Accumulate data for the ConversationPredictor
        self._emotion_probs_history: list[list[float]] = []
        self._vad_history: list[list[float]] = []
        self._confidence_history: list[float] = []
        self._forward_predictions: Optional[ForwardPrediction] = None
        
        # Escalation onset callbacks
        self._onset_callbacks: list = []
        self._onset_threshold: float = 0.6
    
    def update(self, prediction: EmotionPrediction) -> ConversationEmotionState:
        """
        Add new emotion reading and compute updated state.
        
        Args:
            prediction: New emotion prediction from model
            
        Returns:
            Updated conversation emotion state
        """
        snapshot = EmotionSnapshot(
            emotion=prediction.primary_emotion,
            confidence=prediction.confidence,
            valence=prediction.valence,
            arousal=prediction.arousal,
            dominance=prediction.dominance,
        )
        
        self._history.append(snapshot)
        self._all_readings.append(snapshot)
        
        # Accumulate data for the predictor
        self._emotion_probs_history.append(
            list(prediction.emotion_probabilities.values())
        )
        self._vad_history.append(
            [prediction.valence, prediction.arousal, prediction.dominance]
        )
        self._confidence_history.append(prediction.confidence)
        
        # Update EMA smoothed values
        alpha = self.smoothing_alpha
        if self._smoothed_valence is None:
            # First reading: initialize to raw value
            self._smoothed_valence = prediction.valence
            self._smoothed_arousal = prediction.arousal
        else:
            self._smoothed_valence = alpha * prediction.valence + (1 - alpha) * self._smoothed_valence
            self._smoothed_arousal = alpha * prediction.arousal + (1 - alpha) * self._smoothed_arousal
        
        # Track consecutive negative readings (resets on positive)
        if prediction.valence < -0.3:
            self._consecutive_negative += 1
            self._negative_count += 1
        else:
            self._consecutive_negative = 0
        
        if prediction.valence < self._peak_negative_valence:
            self._peak_negative_valence = prediction.valence
        
        # Compute forward predictions if predictor is available
        self._forward_predictions = self._get_forward_predictions()
        
        # Fire onset callbacks if escalation onset detected
        if (
            self._forward_predictions is not None
            and self._forward_predictions.escalation_onset > self._onset_threshold
            and self._onset_callbacks
        ):
            for callback in self._onset_callbacks:
                try:
                    callback(self._forward_predictions, snapshot)
                except Exception:
                    pass  # Don't let callback errors break tracking
        
        return self.get_state()
    
    def _get_forward_predictions(self) -> Optional[ForwardPrediction]:
        """Get forward-looking predictions from the ConversationPredictor."""
        if self._predictor is None:
            return None
        return self._predictor.predict(
            self._emotion_probs_history,
            self._vad_history,
            self._confidence_history,
        )
    
    @staticmethod
    def _map_escalation_risk(will_escalate: float) -> EscalationRisk:
        """Map predictor's will_escalate probability to EscalationRisk enum."""
        if will_escalate > 0.8:
            return EscalationRisk.CRITICAL
        elif will_escalate > 0.5:
            return EscalationRisk.HIGH
        elif will_escalate > 0.3:
            return EscalationRisk.MEDIUM
        return EscalationRisk.LOW

    @staticmethod
    def _map_recommended_tone(tone: RecommendedTone) -> AgentTone:
        """Map predictor's RecommendedTone to AgentTone (same values)."""
        return AgentTone(tone.value)

    def on_escalation_onset(self, callback, threshold: float = 0.6):
        """
        Register a callback fired when escalation onset is detected.
        
        The callback receives (forward_prediction, current_snapshot) and is
        called when escalation_onset probability crosses the threshold.
        Use this to trigger de-escalation workflows in real-time.
        
        Args:
            callback: Function(ForwardPrediction, EmotionSnapshot) -> None
            threshold: Onset probability threshold (default 0.6)
        """
        self._onset_callbacks.append(callback)
        self._onset_threshold = threshold

    def get_state(self) -> ConversationEmotionState:
        """Get current aggregated emotional state."""
        if not self._history:
            return self._empty_state()
        
        current = self._history[-1]
        smoothed_valence = self._smoothed_valence if self._smoothed_valence is not None else 0.0
        smoothed_arousal = self._smoothed_arousal if self._smoothed_arousal is not None else 0.5
        
        # Detect spike: raw reading deviates significantly from smoothed
        spike_detected = abs(current.valence - smoothed_valence) > self.spike_deviation_threshold
        
        # Compute averages
        avg_valence = sum(s.valence for s in self._history) / len(self._history)
        avg_arousal = sum(s.arousal for s in self._history) / len(self._history)
        
        # Compute trajectory
        trajectory, valence_trend = self._compute_trajectory()
        
        # Compute escalation risk using SMOOTHED values (not raw current)
        escalation_risk = self._compute_escalation_risk(current, smoothed_valence)
        
        # Determine recommended tone using SMOOTHED values
        recommended_tone = self._recommend_tone(current, trajectory, escalation_risk)
        
        # Override with predictor results when available
        forward_preds = getattr(self, "_forward_predictions", None)
        if forward_preds is not None:
            escalation_risk = self._map_escalation_risk(forward_preds.will_escalate)
            recommended_tone = self._map_recommended_tone(forward_preds.recommended_tone)
        
        # Generate coaching hint (uses potentially-overridden escalation_risk)
        coaching_hint = self._generate_coaching_hint(
            current, trajectory, escalation_risk, recommended_tone,
            spike_detected=spike_detected,
        )
        
        return ConversationEmotionState(
            current_emotion=current.emotion,
            current_confidence=current.confidence,
            current_valence=current.valence,
            current_arousal=current.arousal,
            current_dominance=current.dominance,
            smoothed_valence=smoothed_valence,
            smoothed_arousal=smoothed_arousal,
            spike_detected=spike_detected,
            consecutive_negative=self._consecutive_negative,
            trajectory=trajectory,
            valence_trend=valence_trend,
            escalation_risk=escalation_risk,
            negative_emotion_count=self._negative_count,
            peak_negative_valence=self._peak_negative_valence,
            avg_valence=avg_valence,
            avg_arousal=avg_arousal,
            segment_count=len(self._all_readings),
            recommended_tone=recommended_tone,
            coaching_hint=coaching_hint,
            forward_predictions=forward_preds,
        )
    
    def _compute_trajectory(self) -> tuple[EmotionTrajectory, float]:
        """Compute emotional trajectory from recent history."""
        if len(self._history) < 3:
            return EmotionTrajectory.STABLE, 0.0
        
        # Compare first half vs second half of window
        mid = len(self._history) // 2
        first_half = list(self._history)[:mid]
        second_half = list(self._history)[mid:]
        
        first_avg = sum(s.valence for s in first_half) / len(first_half)
        second_avg = sum(s.valence for s in second_half) / len(second_half)
        
        trend = second_avg - first_avg
        
        # Check for volatility (high variance)
        valences = [s.valence for s in self._history]
        variance = sum((v - sum(valences)/len(valences))**2 for v in valences) / len(valences)
        
        if variance > 0.15:
            return EmotionTrajectory.VOLATILE, trend
        elif trend > self.trajectory_threshold:
            return EmotionTrajectory.IMPROVING, trend
        elif trend < -self.trajectory_threshold:
            return EmotionTrajectory.DECLINING, trend
        else:
            return EmotionTrajectory.STABLE, trend
    
    def _compute_escalation_risk(
        self, current: EmotionSnapshot, smoothed_valence: float
    ) -> EscalationRisk:
        """
        Assess escalation risk based on smoothed emotion patterns.
        
        Uses smoothed valence (not raw) and requires sustained negative
        readings before escalating to HIGH/CRITICAL. A single angry outburst
        or sarcastic remark won't trigger CRITICAL -- only sustained patterns.
        """
        consecutive = self._consecutive_negative
        threshold = self.escalation_consecutive_threshold
        
        # CRITICAL: sustained high arousal + negative smoothed valence
        # Requires consecutive negative readings to confirm it's not a spike
        if (
            smoothed_valence < -0.5
            and self._smoothed_arousal is not None
            and self._smoothed_arousal > 0.7
            and consecutive >= threshold
        ):
            return EscalationRisk.CRITICAL
        
        # HIGH: sustained negative smoothed valence
        if smoothed_valence < self.escalation_valence_threshold and consecutive >= threshold:
            return EscalationRisk.HIGH
        
        # MEDIUM: smoothed valence is negative but not yet sustained,
        # or a single negative reading (informational, not alarming)
        if smoothed_valence < -0.3 or current.valence < -0.3:
            return EscalationRisk.MEDIUM
        
        return EscalationRisk.LOW
    
    def _recommend_tone(
        self,
        current: EmotionSnapshot,
        trajectory: EmotionTrajectory,
        escalation_risk: EscalationRisk,
    ) -> AgentTone:
        """Recommend agent tone based on user's emotional state."""
        
        # Critical escalation: stay calm, de-escalate
        if escalation_risk == EscalationRisk.CRITICAL:
            return AgentTone.CALM
        
        # Angry user: calm and professional
        if current.emotion == EmotionLabel.ANGRY:
            return AgentTone.CALM
        
        # Sad user: empathetic
        if current.emotion == EmotionLabel.SAD:
            return AgentTone.EMPATHETIC
        
        # Fearful/anxious: reassuring
        if current.emotion == EmotionLabel.FEARFUL:
            return AgentTone.REASSURING
        
        # Happy user: match enthusiasm
        if current.emotion == EmotionLabel.HAPPY:
            return AgentTone.ENTHUSIASTIC
        
        # Declining trajectory: empathetic
        if trajectory == EmotionTrajectory.DECLINING:
            return AgentTone.EMPATHETIC
        
        # Default
        return AgentTone.PROFESSIONAL
    
    def _generate_coaching_hint(
        self,
        current: EmotionSnapshot,
        trajectory: EmotionTrajectory,
        escalation_risk: EscalationRisk,
        tone: AgentTone,
        spike_detected: bool = False,
    ) -> str:
        """Generate coaching hint for the agent."""
        
        hints = []
        
        # Spike-aware coaching: don't overreact to transient readings
        if spike_detected and escalation_risk not in (EscalationRisk.HIGH, EscalationRisk.CRITICAL):
            hints.append("Momentary tone shift detected -- monitoring. No action needed yet.")
        elif escalation_risk == EscalationRisk.CRITICAL:
            hints.append("Customer is highly agitated (sustained). Acknowledge their frustration, stay calm, and focus on resolution.")
        elif escalation_risk == EscalationRisk.HIGH:
            hints.append("Escalation risk is elevated (sustained pattern). Show empathy and work toward a solution.")
        elif escalation_risk == EscalationRisk.MEDIUM:
            if self._consecutive_negative < self.escalation_consecutive_threshold:
                hints.append("Slight negative sentiment noted. Continue monitoring.")
            else:
                hints.append("Negative sentiment building. Consider addressing concerns proactively.")
        
        if trajectory == EmotionTrajectory.DECLINING:
            hints.append("Sentiment is declining. Consider checking in on their concerns.")
        elif trajectory == EmotionTrajectory.IMPROVING:
            hints.append("Good progress -- customer sentiment is improving.")
        
        if current.emotion == EmotionLabel.SAD and not spike_detected:
            hints.append("Customer sounds down. Use a warm, supportive tone.")
        elif current.emotion == EmotionLabel.FEARFUL and not spike_detected:
            hints.append("Customer seems anxious. Provide reassurance and clear information.")
        
        return " ".join(hints) if hints else "Continue with professional, helpful tone."
    
    def _empty_state(self) -> ConversationEmotionState:
        """Return empty/default state."""
        return ConversationEmotionState(
            current_emotion=EmotionLabel.NEUTRAL,
            current_confidence=0.0,
            current_valence=0.0,
            current_arousal=0.5,
            current_dominance=0.5,
            smoothed_valence=0.0,
            smoothed_arousal=0.5,
            spike_detected=False,
            consecutive_negative=0,
            trajectory=EmotionTrajectory.STABLE,
            valence_trend=0.0,
            escalation_risk=EscalationRisk.LOW,
            negative_emotion_count=0,
            peak_negative_valence=0.0,
            avg_valence=0.0,
            avg_arousal=0.5,
            segment_count=0,
            recommended_tone=AgentTone.PROFESSIONAL,
            coaching_hint="No emotion data yet.",
            forward_predictions=None,
        )
    
    def reset(self):
        """Reset tracker for new conversation."""
        self._history.clear()
        self._all_readings.clear()
        self._negative_count = 0
        self._peak_negative_valence = 0.0
        self._smoothed_valence = None
        self._smoothed_arousal = None
        self._consecutive_negative = 0
        self._emotion_probs_history.clear()
        self._vad_history.clear()
        self._confidence_history.clear()


# --- TTS Emotion Mapping ---

# Maps recommended agent tone to TTS emotion tag (for Orpheus)
TONE_TO_TTS_EMOTION = {
    AgentTone.EMPATHETIC: "sad",       # Warm, softer voice
    AgentTone.CALM: "neutral",          # Steady, even
    AgentTone.ENTHUSIASTIC: "happy",    # Upbeat
    AgentTone.PROFESSIONAL: "neutral",  # Clear, standard
    AgentTone.REASSURING: "neutral",    # Confident (neutral with slower pace)
    AgentTone.APOLOGETIC: "sad",        # Softer, sorry tone
}

# TTS speed adjustments for different tones
TONE_TO_TTS_SPEED = {
    AgentTone.EMPATHETIC: 0.9,      # Slightly slower
    AgentTone.CALM: 0.85,           # Slower, deliberate
    AgentTone.ENTHUSIASTIC: 1.1,    # Slightly faster
    AgentTone.PROFESSIONAL: 1.0,    # Normal
    AgentTone.REASSURING: 0.95,     # Slightly slower
    AgentTone.APOLOGETIC: 0.9,      # Slightly slower
}


def get_tts_params_for_tone(tone: AgentTone) -> dict:
    """
    Get TTS parameters for the recommended agent tone.
    
    Returns:
        Dict with 'emotion' and 'speed' for TTS config
    """
    return {
        "emotion": TONE_TO_TTS_EMOTION.get(tone, "neutral"),
        "speed": TONE_TO_TTS_SPEED.get(tone, 1.0),
    }


# --- LLM Context Formatting ---

def format_emotion_context_for_llm(state: ConversationEmotionState) -> str:
    """
    Format emotional state as context for LLM system prompt.
    
    This enables the LLM to generate emotionally-aware responses.
    
    Returns:
        Formatted string to inject into LLM system prompt
    """
    # Use smoothed values for LLM context (more stable, less reactive to spikes)
    valence = state.smoothed_valence
    arousal = state.smoothed_arousal
    
    context_parts = [
        "## Customer Emotional State",
        "",
        f"- **Current emotion**: {state.current_emotion.value} (confidence: {state.current_confidence:.0%})",
        f"- **Sentiment**: {'positive' if valence > 0.2 else 'negative' if valence < -0.2 else 'neutral'} (valence: {valence:+.2f})",
        f"- **Energy level**: {'high' if arousal > 0.6 else 'low' if arousal < 0.4 else 'moderate'}",
        f"- **Trajectory**: {state.trajectory.value}",
        f"- **Escalation risk**: {state.escalation_risk.value}",
    ]
    
    if state.spike_detected:
        context_parts.append(f"- **Note**: Momentary tone shift detected (may be transient, not sustained)")
    
    context_parts.extend([
        "",
        "## Response Guidance",
        "",
        f"- **Recommended tone**: {state.recommended_tone.value}",
        f"- **Coaching**: {state.coaching_hint}",
    ])
    
    # Add specific guidance based on state
    if state.escalation_risk in [EscalationRisk.HIGH, EscalationRisk.CRITICAL]:
        context_parts.extend([
            "",
            "**IMPORTANT**: Customer is at risk of escalation. Prioritize:",
            "1. Acknowledge their frustration explicitly",
            "2. Apologize for any inconvenience",
            "3. Focus on immediate resolution",
            "4. Avoid defensive language",
        ])
    
    if state.trajectory == EmotionTrajectory.DECLINING:
        context_parts.extend([
            "",
            "Note: Customer sentiment has been declining. Consider asking if there's something else bothering them.",
        ])
    
    return "\n".join(context_parts)


def format_emotion_context_compact(state: ConversationEmotionState) -> str:
    """
    Compact emotion context for token-limited LLMs.
    
    Returns:
        Single-line context string
    """
    risk_flag = ""
    if state.escalation_risk in [EscalationRisk.HIGH, EscalationRisk.CRITICAL]:
        risk_flag = "ALERT: "
    spike_note = " [momentary spike]" if state.spike_detected else ""
    return (
        f"{risk_flag}Customer: {state.current_emotion.value} "
        f"({state.smoothed_valence:+.1f} valence, {state.trajectory.value}){spike_note}. "
        f"Use {state.recommended_tone.value} tone."
    )
