"""
Outcome-to-label mapping for fine-tuning.

Converts real-world outcomes into training signals for:
1. Per-utterance ProsodySSM (corrected emotion labels + sample weights)
2. ConversationPredictor (session outcomes → forward-looking head targets)
"""

from dataclasses import dataclass, field
from typing import Optional

from prosody_ssm.model import EmotionLabel


# ==============================================================================
# Per-Utterance Mapping
# ==============================================================================


@dataclass
class UtteranceFeedbackSample:
    """Training sample derived from prediction feedback."""

    prediction_id: str

    # Features (logged at prediction time, no re-extraction needed)
    prosody_features: list[float]
    phonetic_features: list[float]

    # Label (original or corrected)
    emotion_label: str
    emotion_idx: int
    vad_targets: Optional[list[float]] = None  # [valence, arousal, dominance]

    # Training weight
    sample_weight: float = 1.0

    # Source
    source: str = "feedback"  # "correction", "outcome_confirmed", "outcome_contradicted"


# Emotion labels list (must match model.EmotionLabel order)
EMOTION_LABELS = [e.value for e in EmotionLabel]


def map_correction_to_sample(
    prediction_log: dict,
    correction: dict,
) -> UtteranceFeedbackSample:
    """
    Map a human correction to a training sample.

    Corrections are the highest-quality signal. Weight = 2.0.
    """
    correct_emotion = correction["correct_emotion"].lower()

    if correct_emotion not in EMOTION_LABELS:
        raise ValueError(f"Unknown emotion: {correct_emotion}")

    vad = None
    if any(
        correction.get(k) is not None
        for k in ["correct_valence", "correct_arousal", "correct_dominance"]
    ):
        vad = [
            correction.get("correct_valence", prediction_log.get("valence", 0.0)),
            correction.get("correct_arousal", prediction_log.get("arousal", 0.5)),
            correction.get("correct_dominance", prediction_log.get("dominance", 0.5)),
        ]

    return UtteranceFeedbackSample(
        prediction_id=prediction_log["prediction_id"],
        prosody_features=prediction_log["prosody_features"],
        phonetic_features=prediction_log["phonetic_features"],
        emotion_label=correct_emotion,
        emotion_idx=EMOTION_LABELS.index(correct_emotion),
        vad_targets=vad,
        sample_weight=2.0,
        source="correction",
    )


def map_outcome_to_sample(
    prediction_log: dict,
    outcome: dict,
) -> Optional[UtteranceFeedbackSample]:
    """
    Map a per-prediction outcome to a training sample.

    Returns None if outcome doesn't provide a clear training signal.

    Weight logic:
    - Outcome confirms prediction: weight 1.5 (reinforce)
    - Outcome contradicts prediction: weight 2.0 + corrected label
    - Ambiguous: weight 1.0 (use original prediction as-is)
    """
    predicted_emotion = prediction_log.get("emotion", "neutral")
    predicted_valence = prediction_log.get("valence", 0.0)
    vertical = outcome.get("vertical", "")

    corrected_emotion = None
    weight = 1.0
    source = "outcome_confirmed"

    # --- Contact Center ---
    if vertical == "contact_center" and outcome.get("actual_csat") is not None:
        csat = outcome["actual_csat"]

        if csat <= 2.0 and predicted_valence > 0.0:
            # Predicted positive but customer was unhappy
            corrected_emotion = "angry" if csat <= 1.5 else "sad"
            weight = 2.0
            source = "outcome_contradicted"
        elif csat >= 4.0 and predicted_valence < -0.3:
            # Predicted negative but customer was happy
            corrected_emotion = "happy" if csat >= 4.5 else "neutral"
            weight = 2.0
            source = "outcome_contradicted"
        elif csat >= 4.0:
            weight = 1.5  # Confirm
        elif csat <= 2.0:
            weight = 1.5  # Confirm negative

    # --- Sales ---
    elif vertical == "sales" and outcome.get("deal_won") is not None:
        if outcome["deal_won"] and predicted_valence < -0.2:
            # Deal won but we predicted negative
            corrected_emotion = "happy"
            weight = 1.5
            source = "outcome_contradicted"
        elif not outcome["deal_won"] and predicted_valence > 0.3:
            # Deal lost but we predicted positive
            weight = 1.5
            source = "outcome_contradicted"
            # Don't override emotion -- could be many reasons for loss
        else:
            weight = 1.3

    # --- Healthcare ---
    elif vertical == "healthcare" and outcome.get("phq_score") is not None:
        phq = outcome["phq_score"]

        if phq >= 15 and predicted_emotion in ("neutral", "happy", "content"):
            # Severe depression but we predicted positive
            corrected_emotion = "sad"
            weight = 2.0
            source = "outcome_contradicted"
        elif phq >= 10 and predicted_valence > 0.0:
            corrected_emotion = "sad"
            weight = 1.5
            source = "outcome_contradicted"
        elif phq < 5:
            weight = 1.3

    # --- Explicit correctness flag ---
    if outcome.get("outcome_correct") is True:
        weight = max(weight, 1.5)
        source = "outcome_confirmed"
    elif outcome.get("outcome_correct") is False and corrected_emotion is None:
        weight = 1.5
        source = "outcome_contradicted"

    # Build sample
    emotion = corrected_emotion or predicted_emotion
    if emotion not in EMOTION_LABELS:
        return None

    return UtteranceFeedbackSample(
        prediction_id=prediction_log["prediction_id"],
        prosody_features=prediction_log["prosody_features"],
        phonetic_features=prediction_log["phonetic_features"],
        emotion_label=emotion,
        emotion_idx=EMOTION_LABELS.index(emotion),
        sample_weight=weight,
        source=source,
    )


# ==============================================================================
# Session-Level Mapping (ConversationPredictor targets)
# ==============================================================================


@dataclass
class ConversationTrainingTargets:
    """
    Ground truth targets for ConversationPredictor, derived from session outcomes.

    Most targets are applied at EVERY timestep (every-step supervision).
    Exception: escalation_onset uses SPARSE labels -- 1.0 only at the
    K timesteps immediately before/during escalation, 0.0 elsewhere.
    Temporal weighting (0.5 to 1.0) is applied in the loss function, not here.
    """

    session_id: str

    # Binary targets (None = not available for this session)
    will_escalate: Optional[float] = None  # 0.0 or 1.0
    escalation_onset: Optional[list[float]] = None  # Per-timestep sparse labels
    churn_risk: Optional[float] = None
    resolution_prob: Optional[float] = None
    deal_close_prob: Optional[float] = None
    intervention_needed: Optional[float] = None

    # Regression targets
    final_csat: Optional[float] = None  # 1.0 to 5.0
    sentiment_forecast: Optional[float] = None  # -1.0 to 1.0

    # Tone target (index into RecommendedTone enum)
    recommended_tone: Optional[int] = None


def map_session_outcome_to_targets(
    session_outcome: dict,
    n_utterances: int = 0,
    onset_window: int = 3,
) -> ConversationTrainingTargets:
    """
    Convert a session outcome into ConversationPredictor training targets.

    Each outcome field maps to a predictive head target.

    Args:
        session_outcome: Session outcome dict from feedback endpoint
        n_utterances: Number of utterances in the conversation (for onset labels)
        onset_window: Number of utterances before escalation to label as onset (default 3)
    """
    targets = ConversationTrainingTargets(
        session_id=session_outcome["session_id"],
    )

    # --- Binary heads ---

    if session_outcome.get("escalated") is not None:
        targets.will_escalate = 1.0 if session_outcome["escalated"] else 0.0

        # Build sparse onset labels: 1.0 only at the last K utterances
        # before escalation, 0.0 elsewhere
        if n_utterances > 0:
            onset_labels = [0.0] * n_utterances
            if session_outcome["escalated"]:
                # Label the last onset_window utterances as escalation onset
                start = max(0, n_utterances - onset_window)
                for i in range(start, n_utterances):
                    onset_labels[i] = 1.0
            targets.escalation_onset = onset_labels

    if session_outcome.get("churned") is not None:
        targets.churn_risk = 1.0 if session_outcome["churned"] else 0.0

    if session_outcome.get("first_call_resolved") is not None:
        targets.resolution_prob = 1.0 if session_outcome["first_call_resolved"] else 0.0

    if session_outcome.get("deal_won") is not None:
        targets.deal_close_prob = 1.0 if session_outcome["deal_won"] else 0.0

    if session_outcome.get("intervention_occurred") is not None:
        targets.intervention_needed = (
            1.0 if session_outcome["intervention_occurred"] else 0.0
        )

    # --- Regression heads ---

    if session_outcome.get("actual_csat") is not None:
        targets.final_csat = float(session_outcome["actual_csat"])

    if session_outcome.get("final_sentiment") is not None:
        targets.sentiment_forecast = float(session_outcome["final_sentiment"])

    # --- Tone: derive from outcome ---
    # If escalation happened, the right tone was probably "calm"
    # If CSAT was high, whatever tone was used worked
    # This is a softer signal -- we infer what tone SHOULD have been used
    if targets.will_escalate == 1.0:
        targets.recommended_tone = 1  # CALM
    elif targets.final_csat is not None:
        if targets.final_csat >= 4.0:
            targets.recommended_tone = 3  # PROFESSIONAL (it worked)
        elif targets.final_csat <= 2.0:
            targets.recommended_tone = 0  # EMPATHETIC (should have been warmer)

    return targets
