"""
ConversationPredictor: Forward-looking predictive model for conversation outcomes.

At every point in a conversation, predicts HOW the conversation will end --
not the current state, but what's coming. Enables proactive intervention
(de-escalate before escalation, adjust tone before sentiment crashes).

Architecture:
    2-layer GRU over a rolling window of per-utterance ProsodySSM outputs.
    Multi-head output predicts final conversation outcomes at every timestep.

Input per timestep (12-dim):
    - emotion probabilities (8)
    - VAD scores (3)
    - confidence (1)

Predictive heads (all forward-looking):
    - will_escalate: P(conversation escalates)
    - final_csat: predicted final CSAT (1-5)
    - churn_risk: P(customer churns within 30 days)
    - resolution_prob: P(first-call resolution)
    - deal_close_prob: P(deal closes) -- sales vertical
    - intervention_needed: P(clinical intervention) -- healthcare vertical
    - sentiment_forecast: predicted final valence (-1 to +1)
    - recommended_tone: 6-class softmax for optimal agent tone
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from prosody_ssm.model import EmotionLabel, EmotionPrediction


class RecommendedTone(str, Enum):
    """Recommended agent tone for optimal conversation outcome."""

    EMPATHETIC = "empathetic"
    CALM = "calm"
    ENTHUSIASTIC = "enthusiastic"
    PROFESSIONAL = "professional"
    REASSURING = "reassuring"
    APOLOGETIC = "apologetic"


TONE_LABELS = list(RecommendedTone)


@dataclass
class ForwardPrediction:
    """Forward-looking predictions for a conversation at a given point."""

    # Escalation & risk
    will_escalate: float  # P(escalation eventually), 0-1
    escalation_onset: float  # P(escalation starting RIGHT NOW), 0-1
    churn_risk: float  # P(churn within 30 days), 0-1

    # Satisfaction & resolution
    final_csat: float  # Predicted final CSAT, 1-5
    resolution_prob: float  # P(first-call resolution), 0-1

    # Vertical-specific
    deal_close_prob: float  # P(deal closes), 0-1 -- sales
    intervention_needed: float  # P(clinical intervention), 0-1 -- healthcare

    # Sentiment
    sentiment_forecast: float  # Predicted final valence, -1 to +1

    # Action
    recommended_tone: RecommendedTone
    tone_confidence: float  # Confidence in tone recommendation

    # Meta
    prediction_confidence: float  # Overall confidence (based on sequence length)
    utterances_seen: int


if TORCH_AVAILABLE:

    class ConversationPredictor(nn.Module):
        """
        Forward-looking conversation outcome predictor.

        Takes a rolling window of per-utterance ProsodySSM outputs and
        predicts final conversation outcomes at every timestep.

        The GRU processes the sequence and each head predicts the final
        outcome from that point forward, enabling early warning signals.
        """

        # Input dimension: 8 emotion probs + 3 VAD + 1 confidence
        INPUT_DIM = 12
        NUM_TONES = len(TONE_LABELS)

        def __init__(
            self,
            input_dim: int = 12,
            hidden_dim: int = 64,
            n_layers: int = 2,
            dropout: float = 0.1,
            max_window: int = 20,
        ):
            super().__init__()

            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.max_window = max_window

            # Input projection: map the 12-dim per-utterance summary
            # (8 emotion probs + 3 VAD + 1 confidence) up to hidden_dim.
            # This is a bottleneck -- the model must learn which of the 12
            # input features are most predictive of conversation outcomes.
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # GRU backbone: processes the conversation as a sequence of utterances.
            # Unidirectional (causal) -- at time t, the GRU only sees utterances 1..t.
            # This is critical for streaming: we predict outcomes BEFORE the conversation ends.
            #
            # Why GRU not Mamba: sequences are short (10-20 utterances). GRU is simpler,
            # faster, and has enough capacity. Mamba's O(n) advantage over O(n^2) attention
            # is irrelevant at this length. GRU's gating (reset/update gates) is well-suited
            # for detecting when a conversation "shifts" -- e.g., customer goes from calm to angry.
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
                bidirectional=False,
            )

            # Shared representation: post-GRU projection shared by all heads.
            # Each head gets the same representation but learns different projections.
            self.shared_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # --- Forward-looking predictive heads ---
            # All heads predict at EVERY timestep. During training, every prediction
            # is supervised against the final conversation outcome (every-step supervision).
            # During inference, later predictions are more accurate but earlier ones
            # provide early warning.

            # Binary heads (sigmoid output, trained with BCE):
            # will_escalate: P(customer escalates at any point) -- same target at all timesteps
            # escalation_onset: P(escalation is starting RIGHT NOW) -- SPARSE labels,
            #   only 1.0 at the K utterances immediately before/during escalation, 0.0 elsewhere.
            #   This detects the inflection point, not the overall trend.
            # churn_risk: P(customer churns within 30 days)
            # resolution_prob: P(issue resolved on this call)
            # deal_close_prob: P(deal closes) -- sales vertical
            # intervention_needed: P(clinical intervention needed) -- healthcare vertical
            self.will_escalate_head = nn.Linear(hidden_dim, 1)
            self.escalation_onset_head = nn.Linear(hidden_dim, 1)
            self.churn_risk_head = nn.Linear(hidden_dim, 1)
            self.resolution_prob_head = nn.Linear(hidden_dim, 1)
            self.deal_close_prob_head = nn.Linear(hidden_dim, 1)
            self.intervention_needed_head = nn.Linear(hidden_dim, 1)

            # Regression heads:
            # final_csat: predicted CSAT at end of conversation (1-5 scale, clamped)
            # sentiment_forecast: predicted final valence (-1 to +1, tanh)
            self.final_csat_head = nn.Linear(hidden_dim, 1)
            self.sentiment_forecast_head = nn.Linear(hidden_dim, 1)

            # Multi-class head:
            # recommended_tone: what agent tone to use RIGHT NOW to steer toward
            # the best outcome. 6 classes: empathetic, calm, enthusiastic,
            # professional, reassuring, apologetic.
            self.tone_head = nn.Linear(hidden_dim, self.NUM_TONES)

            self._init_weights()

        def _init_weights(self):
            """Initialize weights with small values for stable training."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            # Initialize binary heads with slight negative bias
            # (most conversations don't escalate / churn)
            for head in [
                self.will_escalate_head,
                self.escalation_onset_head,
                self.churn_risk_head,
                self.intervention_needed_head,
            ]:
                nn.init.constant_(head.bias, -1.0)

            # Resolution is usually positive
            nn.init.constant_(self.resolution_prob_head.bias, 0.5)

            # CSAT centered at 3.0
            nn.init.constant_(self.final_csat_head.bias, 3.0)

        def forward(
            self,
            utterance_features: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
        ) -> dict[str, torch.Tensor]:
            """
            Forward pass: predict outcomes at every timestep.

            Args:
                utterance_features: (batch, seq_len, input_dim)
                    Per-utterance vectors: [emotion_probs(8), VAD(3), confidence(1)]
                lengths: (batch,) actual sequence lengths (for masking)

            Returns:
                Dict of predictions, each shaped (batch, seq_len) or (batch, seq_len, n_classes).
                Every timestep t contains the model's prediction of the FINAL
                conversation outcome given utterances 1..t.
            """
            batch_size, seq_len, _ = utterance_features.shape

            # Project input
            x = self.input_proj(utterance_features)  # (B, T, H)

            # Pack if lengths provided (for efficiency with variable-length sequences)
            if lengths is not None:
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )

            # GRU processes causally: output at t only sees 1..t
            gru_out, _ = self.gru(x)  # (B, T, H)

            if lengths is not None:
                gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                    gru_out, batch_first=True, total_length=seq_len
                )

            # Shared representation
            h = self.shared_head(gru_out)  # (B, T, H)

            # Compute all heads at every timestep
            predictions = {
                # Binary (sigmoid) -- shape (B, T)
                "will_escalate": torch.sigmoid(
                    self.will_escalate_head(h).squeeze(-1)
                ),
                "escalation_onset": torch.sigmoid(
                    self.escalation_onset_head(h).squeeze(-1)
                ),
                "churn_risk": torch.sigmoid(
                    self.churn_risk_head(h).squeeze(-1)
                ),
                "resolution_prob": torch.sigmoid(
                    self.resolution_prob_head(h).squeeze(-1)
                ),
                "deal_close_prob": torch.sigmoid(
                    self.deal_close_prob_head(h).squeeze(-1)
                ),
                "intervention_needed": torch.sigmoid(
                    self.intervention_needed_head(h).squeeze(-1)
                ),
                # Regression -- shape (B, T)
                "final_csat": self.final_csat_head(h).squeeze(-1).clamp(1.0, 5.0),
                "sentiment_forecast": torch.tanh(
                    self.sentiment_forecast_head(h).squeeze(-1)
                ),
                # Multi-class -- shape (B, T, NUM_TONES)
                "tone_logits": self.tone_head(h),
            }

            return predictions

        def predict_step(
            self,
            utterance_features: torch.Tensor,
            hidden: Optional[torch.Tensor] = None,
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
            """
            Single-step prediction for streaming inference.

            Args:
                utterance_features: (batch, 1, input_dim) single utterance
                hidden: (n_layers, batch, hidden_dim) GRU hidden state

            Returns:
                predictions: dict of predictions for this timestep
                hidden: updated GRU hidden state
            """
            x = self.input_proj(utterance_features)  # (B, 1, H)
            gru_out, hidden = self.gru(x, hidden)  # (B, 1, H)
            h = self.shared_head(gru_out)  # (B, 1, H)

            predictions = {
                "will_escalate": torch.sigmoid(
                    self.will_escalate_head(h).squeeze(-1)
                ).squeeze(-1),
                "escalation_onset": torch.sigmoid(
                    self.escalation_onset_head(h).squeeze(-1)
                ).squeeze(-1),
                "churn_risk": torch.sigmoid(
                    self.churn_risk_head(h).squeeze(-1)
                ).squeeze(-1),
                "resolution_prob": torch.sigmoid(
                    self.resolution_prob_head(h).squeeze(-1)
                ).squeeze(-1),
                "deal_close_prob": torch.sigmoid(
                    self.deal_close_prob_head(h).squeeze(-1)
                ).squeeze(-1),
                "intervention_needed": torch.sigmoid(
                    self.intervention_needed_head(h).squeeze(-1)
                ).squeeze(-1),
                "final_csat": self.final_csat_head(h).squeeze(-1).squeeze(-1).clamp(1.0, 5.0),
                "sentiment_forecast": torch.tanh(
                    self.sentiment_forecast_head(h).squeeze(-1)
                ).squeeze(-1),
                "tone_logits": self.tone_head(h).squeeze(1),  # (B, NUM_TONES)
            }

            return predictions, hidden

        def predict(
            self,
            emotion_probs: list[list[float]],
            vad_scores: list[list[float]],
            confidences: list[float],
        ) -> ForwardPrediction:
            """
            High-level prediction from a conversation history.

            Args:
                emotion_probs: List of emotion probability vectors per utterance
                vad_scores: List of [valence, arousal, dominance] per utterance
                confidences: List of confidence scores per utterance

            Returns:
                ForwardPrediction with all forward-looking predictions
            """
            self.eval()

            # Build input tensor
            n_utterances = len(emotion_probs)
            features = []
            for i in range(n_utterances):
                step = emotion_probs[i] + vad_scores[i] + [confidences[i]]
                features.append(step)

            x = torch.tensor([features], dtype=torch.float32)  # (1, T, 12)

            with torch.no_grad():
                preds = self.forward(x)

            # Take the LAST timestep (most informed prediction)
            t = n_utterances - 1

            tone_probs = F.softmax(preds["tone_logits"][0, t], dim=-1)
            tone_idx = tone_probs.argmax().item()

            # Confidence increases with sequence length
            pred_confidence = min(1.0, 0.3 + 0.7 * (n_utterances / self.max_window))

            return ForwardPrediction(
                will_escalate=float(preds["will_escalate"][0, t]),
                escalation_onset=float(preds["escalation_onset"][0, t]),
                churn_risk=float(preds["churn_risk"][0, t]),
                final_csat=float(preds["final_csat"][0, t]),
                resolution_prob=float(preds["resolution_prob"][0, t]),
                deal_close_prob=float(preds["deal_close_prob"][0, t]),
                intervention_needed=float(preds["intervention_needed"][0, t]),
                sentiment_forecast=float(preds["sentiment_forecast"][0, t]),
                recommended_tone=TONE_LABELS[tone_idx],
                tone_confidence=float(tone_probs[tone_idx]),
                prediction_confidence=pred_confidence,
                utterances_seen=n_utterances,
            )

        @classmethod
        def from_pretrained(cls, model_path: str) -> "ConversationPredictor":
            """Load a pretrained model."""
            checkpoint = torch.load(model_path, map_location="cpu")
            config = checkpoint.get("config", {})
            model = cls(**config)
            model.load_state_dict(checkpoint["model_state_dict"])
            return model

        def save_pretrained(self, model_path: str, config: Optional[dict] = None):
            """Save model to path."""
            if config is None:
                config = {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "max_window": self.max_window,
                }
            torch.save(
                {"model_state_dict": self.state_dict(), "config": config},
                model_path,
            )

    def compute_conversation_loss(
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        lengths: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """
        Compute multi-task loss with every-step supervision and temporal weighting.

        Each head's prediction at every timestep is supervised against the
        final conversation outcome. Later timesteps get higher weight (the
        model is more informed and should be more accurate).

        Args:
            predictions: dict from ConversationPredictor.forward()
            targets: dict with ground truth for each head
                Binary targets: (batch,) float 0/1
                Regression targets: (batch,) float
                Tone target: (batch,) long class index
            lengths: (batch,) actual sequence lengths
            max_seq_len: maximum sequence length in batch

        Returns:
            Scalar loss tensor
        """
        batch_size = lengths.shape[0]
        device = lengths.device

        # Build temporal weights: 0.5 at step 1, 1.0 at final step
        # Shape: (batch, max_seq_len)
        step_weights = torch.zeros(batch_size, max_seq_len, device=device)
        for i in range(batch_size):
            T = lengths[i].item()
            if T > 0:
                t = torch.arange(1, T + 1, dtype=torch.float32, device=device)
                step_weights[i, :T] = 0.5 + 0.5 * (t / T)

        # Build padding mask
        mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.float()  # (B, T)

        total_loss = torch.tensor(0.0, device=device)
        n_heads = 0

        # --- Binary heads (BCE) ---
        binary_heads = [
            "will_escalate",
            "escalation_onset",
            "churn_risk",
            "resolution_prob",
            "deal_close_prob",
            "intervention_needed",
        ]

        for head_name in binary_heads:
            if head_name not in targets:
                continue

            pred = predictions[head_name]  # (B, T)
            target = targets[head_name].unsqueeze(1).expand_as(pred)  # (B,) -> (B, T)

            # Per-step BCE
            bce = F.binary_cross_entropy(pred, target, reduction="none")  # (B, T)
            weighted = bce * step_weights * mask
            head_loss = weighted.sum() / mask.sum().clamp(min=1)

            total_loss = total_loss + head_loss
            n_heads += 1

        # --- Regression heads (MSE) ---
        regression_heads = [
            ("final_csat", 1.0),
            ("sentiment_forecast", 1.0),
        ]

        for head_name, scale in regression_heads:
            if head_name not in targets:
                continue

            pred = predictions[head_name]  # (B, T)
            target = targets[head_name].unsqueeze(1).expand_as(pred)  # (B,) -> (B, T)

            mse = F.mse_loss(pred, target, reduction="none")  # (B, T)
            weighted = mse * step_weights * mask * scale
            head_loss = weighted.sum() / mask.sum().clamp(min=1)

            total_loss = total_loss + head_loss
            n_heads += 1

        # --- Tone head (CE) ---
        if "recommended_tone" in targets:
            tone_logits = predictions["tone_logits"]  # (B, T, NUM_TONES)
            tone_target = targets["recommended_tone"]  # (B,) long

            B, T, C = tone_logits.shape
            # Expand target to every timestep
            tone_target_expanded = tone_target.unsqueeze(1).expand(B, T)  # (B, T)

            # Reshape for cross_entropy
            ce = F.cross_entropy(
                tone_logits.reshape(B * T, C),
                tone_target_expanded.reshape(B * T),
                reduction="none",
            ).reshape(B, T)

            weighted = ce * step_weights * mask
            head_loss = weighted.sum() / mask.sum().clamp(min=1)

            total_loss = total_loss + head_loss
            n_heads += 1

        # Average across active heads
        if n_heads > 0:
            total_loss = total_loss / n_heads

        return total_loss

else:
    # Fallback when PyTorch not available
    class ConversationPredictor:
        """Heuristic fallback when PyTorch is not available."""

        def __init__(self, **kwargs):
            pass

        def predict(
            self,
            emotion_probs: list[list[float]],
            vad_scores: list[list[float]],
            confidences: list[float],
        ) -> ForwardPrediction:
            """Simple heuristic-based forward prediction."""
            n = len(vad_scores)
            if n == 0:
                return self._empty_prediction()

            # Use last few utterances for heuristics
            recent_valence = [v[0] for v in vad_scores[-5:]]
            recent_arousal = [v[1] for v in vad_scores[-5:]]
            avg_valence = sum(recent_valence) / len(recent_valence)
            avg_arousal = sum(recent_arousal) / len(recent_arousal)
            avg_confidence = sum(confidences[-5:]) / len(confidences[-5:])

            # Heuristic escalation risk
            will_escalate = max(0.0, min(1.0, (-avg_valence + avg_arousal) / 2))

            # Heuristic CSAT from valence
            final_csat = max(1.0, min(5.0, 3.0 + avg_valence * 2.0))

            # Heuristic tone
            if will_escalate > 0.6:
                tone = RecommendedTone.CALM
            elif avg_valence < -0.3:
                tone = RecommendedTone.EMPATHETIC
            elif avg_valence > 0.3:
                tone = RecommendedTone.ENTHUSIASTIC
            else:
                tone = RecommendedTone.PROFESSIONAL

            # Heuristic onset: escalation starting if recent sharp valence drop
            onset = 0.0
            if n >= 3:
                recent_v = recent_valence[-3:]
                if len(recent_v) >= 3 and recent_v[-1] < -0.3 and recent_v[-1] < recent_v[0] - 0.3:
                    onset = min(1.0, abs(recent_v[-1] - recent_v[0]))

            return ForwardPrediction(
                will_escalate=will_escalate,
                escalation_onset=onset,
                churn_risk=max(0.0, will_escalate * 0.5),
                final_csat=final_csat,
                resolution_prob=max(0.0, min(1.0, 1.0 - will_escalate)),
                deal_close_prob=max(0.0, min(1.0, (avg_valence + 1) / 2)),
                intervention_needed=max(0.0, min(1.0, -avg_valence * avg_arousal)),
                sentiment_forecast=avg_valence,
                recommended_tone=tone,
                tone_confidence=avg_confidence,
                prediction_confidence=min(1.0, 0.3 + 0.7 * (n / 20)),
                utterances_seen=n,
            )

        def _empty_prediction(self) -> ForwardPrediction:
            return ForwardPrediction(
                will_escalate=0.0,
                escalation_onset=0.0,
                churn_risk=0.0,
                final_csat=3.0,
                resolution_prob=0.5,
                deal_close_prob=0.5,
                intervention_needed=0.0,
                sentiment_forecast=0.0,
                recommended_tone=RecommendedTone.PROFESSIONAL,
                tone_confidence=0.0,
                prediction_confidence=0.0,
                utterances_seen=0,
            )

        @classmethod
        def from_pretrained(cls, model_path: str) -> "ConversationPredictor":
            return cls()

        def save_pretrained(self, model_path: str, config=None):
            pass
