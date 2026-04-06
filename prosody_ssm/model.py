"""
ProsodySSM: State Space Model for prosody analysis and KPI outcome prediction.

We pretrain ProsodySSM on public emotion datasets to learn prosodic
representations, then adapt to client-defined KPIs via a KPI-conditioned
head when outcome labels are available.

Architecture:
    1. Feature backbone:
       - WavLM-Large (frozen) for rich prosodic features from raw waveforms
       - Legacy MFCC/prosodic embeddings path for backward compat / CPU inference
    2. Stack of Mamba blocks (mamba_ssm selective scan, or S4D fallback)
    3. Emotion classification head (pretraining — learns prosodic representations)
    4. VAD regression head (prosodic state: valence, arousal, dominance)
    5. KPI-conditioned prediction head (downstream — predicts client-defined KPI
       outcomes, requires outcome supervision)

Training stages:
    Stage 1 (pretraining): Train emotion + VAD heads on public emotion corpora
        (IEMOCAP, RAVDESS, etc.) to learn rich prosodic representations in
        the SSM backbone. This gives us a general-purpose prosody encoder.

    Stage 2 (adaptation): Freeze or fine-tune the backbone, train the
        KPI-conditioned head on client outcome data. Each training sample is
        (prosody_features, kpi_config, actual_kpi_value). The KPI head is
        conditioned on KPI metadata (type, direction, range) so a single
        model serves all clients and KPI types.

The emotion head remains available for representation learning and as a
useful signal where outcome data is not yet available.
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

from prosody_ssm.features import PhoneticFeatures, ProsodyFeatures

# ---------------------------------------------------------------------------
# Emotion types (legacy — kept for backward compatibility)
# ---------------------------------------------------------------------------


class EmotionLabel(str, Enum):
    """Emotion labels used for pretraining on public emotion corpora."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    CONTEMPT = "contempt"


@dataclass
class EmotionPrediction:
    """Result of emotion classification (pretraining head)."""
    primary_emotion: EmotionLabel
    confidence: float
    emotion_probabilities: dict[str, float]
    valence: float
    arousal: float
    dominance: float


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class ProsodyPrediction:
    """
    Prosodic state prediction from the SSM backbone.

    These are continuous signal dimensions.
    They describe the speaker's vocal state in a dimensionless space.
    """

    valence: float      # -1 (negative tone) to +1 (positive tone)
    arousal: float      # 0 (calm) to 1 (activated)
    dominance: float    # 0 (submissive) to 1 (dominant)
    confidence: float   # model confidence in the prediction


@dataclass
class KPIModelConfig:
    """
    KPI definition passed to the model as conditioning input.

    The model treats this as "what are we predicting?" configuration.
    This mirrors the Kpi table in the shared database but encoded
    as tensors for the model.
    """

    kpi_type: int           # 0=scalar, 1=binary, 2=categorical
    direction: int          # 0=higher_is_better, 1=lower_is_better
    range_min: float = 0.0
    range_max: float = 1.0
    n_categories: int = 0   # number of categories (for categorical KPIs)


@dataclass
class KPIModelOutput:
    """Single KPI prediction from the model."""

    value: float            # predicted value (scalar), probability (binary)
    confidence: float       # prediction confidence
    category_logits: Optional[list[float]] = None  # for categorical KPIs


# ---------------------------------------------------------------------------
# SSM backbone components
# ---------------------------------------------------------------------------


if TORCH_AVAILABLE:

    # Try mamba_ssm for real Mamba selective scan
    MAMBA_SSM_AVAILABLE = False
    try:
        from mamba_ssm import Mamba
        MAMBA_SSM_AVAILABLE = True
    except ImportError:
        pass

    # ---------------------------------------------------------------
    # S4D fallback (only defined when mamba_ssm is not installed)
    # ---------------------------------------------------------------

    if not MAMBA_SSM_AVAILABLE:

        class _S4DKernel(nn.Module):
            """
            S4D kernel: Structured State Space with Diagonal matrices.

            Fallback implementation used when mamba_ssm is not installed.
            """

            def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
                super().__init__()
                self.d_model = d_model
                self.d_state = d_state

                # S4D-Lin initialization: negative real part (stability), imaginary
                # part spaces frequencies across the spectrum.
                A_real = -torch.ones(d_state) * 0.5
                A_imag = torch.arange(d_state, dtype=torch.float32) * np.pi
                self.register_buffer('A_real', A_real)
                self.register_buffer('A_imag', A_imag)

                self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
                self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

                log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
                self.log_dt = nn.Parameter(log_dt)
                self.D = nn.Parameter(torch.ones(d_model))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch, length, _ = x.shape
                dt = torch.exp(self.log_dt)

                A_bar = torch.exp(
                    (-self.A_real.unsqueeze(0) + 1j * self.A_imag.unsqueeze(0)) * dt.unsqueeze(1)
                )
                B_bar = dt.unsqueeze(1) * self.B

                h = torch.zeros(batch, self.d_model, self.d_state, dtype=torch.complex64, device=x.device)
                outputs = []

                for t in range(length):
                    h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x[:, t, :].unsqueeze(-1)
                    y = torch.sum(self.C.unsqueeze(0) * h, dim=-1).real + self.D * x[:, t, :]
                    outputs.append(y)

                return torch.stack(outputs, dim=1)

            def step(
                self,
                x: torch.Tensor,
                h: Optional[torch.Tensor] = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Single-step recurrent forward for streaming."""
                batch = x.shape[0]
                dt = torch.exp(self.log_dt)

                A_bar = torch.exp(
                    (-self.A_real.unsqueeze(0) + 1j * self.A_imag.unsqueeze(0)) * dt.unsqueeze(1)
                )
                B_bar = dt.unsqueeze(1) * self.B

                if h is None:
                    h = torch.zeros(batch, self.d_model, self.d_state, dtype=torch.complex64, device=x.device)

                h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x.unsqueeze(-1)
                y = torch.sum(self.C.unsqueeze(0) * h, dim=-1).real + self.D * x

                return y, h

    # ---------------------------------------------------------------
    # Mamba block
    # ---------------------------------------------------------------

    if MAMBA_SSM_AVAILABLE:

        class MambaBlock(nn.Module):
            """Mamba selective state space block (mamba_ssm backend)."""

            def __init__(
                self,
                d_model: int,
                d_state: int = 64,
                d_conv: int = 4,
                expand: int = 2,
                dropout: float = 0.1,
            ):
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = x
                x = self.norm(x)
                x = self.mamba(x)
                x = self.dropout(x)
                return x + residual

    else:

        class MambaBlock(nn.Module):
            """Mamba-style selective state space block (S4D fallback)."""

            def __init__(
                self,
                d_model: int,
                d_state: int = 64,
                d_conv: int = 4,
                expand: int = 2,
                dropout: float = 0.1,
            ):
                super().__init__()
                self.d_model = d_model
                self.d_inner = d_model * expand

                self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
                self.conv1d = nn.Conv1d(
                    self.d_inner, self.d_inner,
                    kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner,
                )
                self.ssm = _S4DKernel(self.d_inner, d_state)
                self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
                self.dropout = nn.Dropout(dropout)
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = x
                x = self.norm(x)

                x_and_gate = self.in_proj(x)
                x_main, gate = x_and_gate.chunk(2, dim=-1)

                x_main = x_main.transpose(1, 2)
                x_main = self.conv1d(x_main)[:, :, :residual.shape[1]]
                x_main = x_main.transpose(1, 2)

                x_main = F.silu(x_main)
                x_main = self.ssm(x_main)
                x_main = x_main * F.silu(gate)

                x_main = self.out_proj(x_main)
                x_main = self.dropout(x_main)

                return x_main + residual

            def step(
                self,
                x: torch.Tensor,
                state: Optional[dict] = None,
            ) -> tuple[torch.Tensor, dict]:
                """Single-step forward for streaming."""
                if state is None:
                    state = {}

                residual = x
                x = self.norm(x)

                x_and_gate = self.in_proj(x)
                x_main, gate = x_and_gate.chunk(2, dim=-1)

                d_conv = self.conv1d.kernel_size[0]
                conv_buf = state.get('conv_buf', torch.zeros(
                    x.shape[0], self.d_inner, d_conv, device=x.device
                ))
                conv_buf = torch.cat([conv_buf[:, :, 1:], x_main.unsqueeze(-1)], dim=-1)
                x_main = self.conv1d(conv_buf)[:, :, -1]

                x_main = F.silu(x_main)

                ssm_h = state.get('ssm_h', None)
                x_main, ssm_h = self.ssm.step(x_main, ssm_h)

                x_main = x_main * F.silu(gate)
                x_main = self.out_proj(x_main)

                new_state = {'ssm_h': ssm_h, 'conv_buf': conv_buf}
                return x_main + residual, new_state

    # -------------------------------------------------------------------
    # WavLM feature backbone
    # -------------------------------------------------------------------

    class _GradientReversalFn(torch.autograd.Function):
        """Flip gradients during backward pass for adversarial training."""

        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.alpha * grad_output, None

    class WavLMBackbone(nn.Module):
        """Frozen WavLM-Large as prosodic feature extractor."""

        def __init__(self, d_model: int = 256, use_weighted_layers: bool = False):
            super().__init__()
            try:
                from transformers import WavLMModel
            except ImportError as e:
                raise ImportError(
                    "WavLMBackbone requires the 'transformers' package. "
                    "Install with: pip install transformers"
                ) from e
            self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
            for param in self.wavlm.parameters():
                param.requires_grad = False

            self.use_weighted_layers = use_weighted_layers
            if use_weighted_layers:
                n_layers = self.wavlm.config.num_hidden_layers + 1  # +1 for embedding layer
                self.layer_weights = nn.Parameter(torch.zeros(n_layers))

            self.proj = nn.Linear(1024, d_model)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            """waveform: (batch, samples) at 16kHz -> (batch, seq_len, d_model)"""
            with torch.no_grad():
                outputs = self.wavlm(waveform, output_hidden_states=self.use_weighted_layers)

            if self.use_weighted_layers:
                hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, 1024)
                stacked = torch.stack(hidden_states, dim=0)  # (n_layers, batch, seq_len, 1024)
                weights = F.softmax(self.layer_weights, dim=0)
                features = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
            else:
                features = outputs.last_hidden_state

            return self.norm(self.proj(features))

    # -------------------------------------------------------------------
    # KPI-conditioned prediction head
    # -------------------------------------------------------------------

    class KPIHead(nn.Module):
        """
        Predicts a single KPI outcome, conditioned on the KPI's configuration.

        The same head handles all KPI types. The KPI config (type, direction,
        range) is encoded as a conditioning vector and concatenated with the
        prosodic representation before prediction.

        This means a single model serves all clients and all KPI definitions.
        The model learns: "given this prosodic state and this KPI schema,
        what is the predicted outcome?"
        """

        N_KPI_TYPES = 3    # scalar, binary, categorical
        N_DIRECTIONS = 2    # higher_is_better, lower_is_better

        def __init__(self, d_model: int, d_kpi_embed: int = 32, max_categories: int = 16):
            super().__init__()

            self.d_kpi_embed = d_kpi_embed
            self.max_categories = max_categories

            # KPI config embeddings -- these encode "what KPI are we predicting?"
            # so one model serves all clients and KPI types.
            #
            # type_embed: {scalar, binary, categorical} -> learned vector
            #   Different KPI types need fundamentally different decision boundaries.
            #   CSAT (scalar) vs churn (binary) vs call disposition (categorical).
            #
            # dir_embed: {higher_is_better, lower_is_better} -> learned vector
            #   Tells the model whether positive prosody (high valence) should
            #   predict a higher or lower KPI value.
            #
            # range_proj: [min, max] -> learned vector
            #   Encodes the scale. CSAT 1-5 vs NPS -100 to +100 vs handle_time 0-3600s.
            #   Without this, the model would conflate all scalar outputs.
            self.type_embed = nn.Embedding(self.N_KPI_TYPES, d_kpi_embed)
            self.dir_embed = nn.Embedding(self.N_DIRECTIONS, d_kpi_embed)
            self.range_proj = nn.Linear(2, d_kpi_embed)

            # Total conditioning dimension = 3 * d_kpi_embed (type + direction + range)
            cond_dim = d_kpi_embed * 3
            input_dim = d_model + cond_dim

            # Shared trunk: takes [prosodic_repr, kpi_conditioning] and produces
            # a KPI-aware representation. The model learns: "given this speaker's
            # vocal state AND this KPI schema, what should the prediction be?"
            self.trunk = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
            )

            # Scalar/binary head: single value output.
            # For scalar KPIs: raw value (clamped to range post-hoc).
            # For binary KPIs: logit (sigmoid applied post-hoc).
            self.value_head = nn.Linear(d_model, 1)

            # Categorical head: logits over up to max_categories classes.
            # Only used when kpi_type == 2 (categorical).
            self.category_head = nn.Linear(d_model, max_categories)

        def forward(
            self,
            prosody_repr: torch.Tensor,
            kpi_type: torch.Tensor,
            kpi_direction: torch.Tensor,
            kpi_range: torch.Tensor,
            n_categories: Optional[torch.Tensor] = None,
        ) -> dict[str, torch.Tensor]:
            """
            Predict KPI outcome conditioned on KPI config.

            Args:
                prosody_repr: (batch, d_model) pooled prosodic representation
                kpi_type: (batch,) int — 0=scalar, 1=binary, 2=categorical
                kpi_direction: (batch,) int — 0=higher_is_better, 1=lower_is_better
                kpi_range: (batch, 2) float — [range_min, range_max]
                n_categories: (batch,) int — for categorical KPIs

            Returns:
                Dict with 'value' (batch,) and optionally 'category_logits' (batch, max_categories)
            """
            # Encode KPI config
            type_emb = self.type_embed(kpi_type)         # (B, d_kpi)
            dir_emb = self.dir_embed(kpi_direction)      # (B, d_kpi)
            range_emb = self.range_proj(kpi_range)       # (B, d_kpi)

            cond = torch.cat([type_emb, dir_emb, range_emb], dim=-1)  # (B, 3*d_kpi)

            # Concatenate prosodic repr with KPI conditioning
            x = torch.cat([prosody_repr, cond], dim=-1)  # (B, d_model + 3*d_kpi)
            x = self.trunk(x)                            # (B, d_model)

            # Value prediction (scalar / binary probability)
            value = self.value_head(x).squeeze(-1)       # (B,)

            result = {"value": value}

            # Category logits (for categorical KPIs)
            result["category_logits"] = self.category_head(x)  # (B, max_categories)

            return result


    # -------------------------------------------------------------------
    # Main model
    # -------------------------------------------------------------------

    class ProsodySSM(nn.Module):
        """
        State Space Model for prosody analysis and KPI outcome prediction.

        Output heads:
            - Emotion classifier (pretraining — public emotion corpora)
            - VAD regression (continuous prosodic state dimensions)
            - Prosodic signal heads (interpreted hidden state)
            - KPI-conditioned prediction (downstream — client outcomes)
        """

        SIGNAL_NAMES = [
            "engagement", "stress", "certainty", "rapport",
            "empathy", "tempo", "intensity", "expressiveness",
        ]

        def __init__(
            self,
            d_model: int = 256,
            n_layers: int = 4,
            d_state: int = 64,
            n_emotions: int = 8,
            d_kpi_embed: int = 32,
            max_categories: int = 16,
            dropout: float = 0.1,
            use_wavlm: bool = True,
            prosody_dim: int = 28,
            phonetic_dim: int = 4,
            use_weighted_wavlm_layers: bool = False,
            use_dual_stream_fusion: bool = False,
            n_speakers: int = 0,
            speaker_grl_alpha: float = 0.1,
            specaugment_prob: float = 0.0,
            specaugment_time_mask: int = 20,
            specaugment_freq_mask: int = 64,
        ):
            super().__init__()

            self.use_wavlm = use_wavlm
            self.d_model = d_model
            self.d_state = d_state
            self.n_emotions = n_emotions
            self.prosody_dim = prosody_dim
            self.phonetic_dim = phonetic_dim
            self.use_weighted_wavlm_layers = use_weighted_wavlm_layers
            self.use_dual_stream_fusion = use_dual_stream_fusion
            self.n_speakers = n_speakers
            self.speaker_grl_alpha = speaker_grl_alpha
            self.specaugment_prob = specaugment_prob
            self.specaugment_time_mask = specaugment_time_mask
            self.specaugment_freq_mask = specaugment_freq_mask

            # --- Feature extraction ---
            if use_wavlm:
                self.backbone = WavLMBackbone(d_model, use_weighted_layers=use_weighted_wavlm_layers)
            else:
                self.prosody_embed = nn.Sequential(
                    nn.Linear(prosody_dim, d_model // 2),
                    nn.LayerNorm(d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.phonetic_embed = nn.Sequential(
                    nn.Linear(phonetic_dim, d_model // 2),
                    nn.LayerNorm(d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                )

            # Dual-stream: fuse WavLM features with MFCC prosody features
            if use_dual_stream_fusion and use_wavlm:
                self.prosody_stream = nn.Sequential(
                    nn.Linear(prosody_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                )
                self.stream_gate = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.Sigmoid(),
                )

            # --- SSM backbone ---
            self.blocks = nn.ModuleList([
                MambaBlock(d_model, d_state, dropout=dropout)
                for _ in range(n_layers)
            ])

            # --- Output heads ---

            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_emotions),
            )

            self.vad_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 3),
                nn.Tanh(),
            )

            self.kpi_head = KPIHead(
                d_model=d_model,
                d_kpi_embed=d_kpi_embed,
                max_categories=max_categories,
            )

            # Speaker adversarial head (gradient reversal for speaker-invariant representations)
            if n_speakers > 0:
                self.speaker_head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, n_speakers),
                )

            # Prosodic signal heads — interpret the SSM hidden state into
            # actionable signals (engagement, stress, certainty, etc.)
            self.signal_heads = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, len(self.SIGNAL_NAMES)),
                nn.Sigmoid(),
            )

            self._init_weights()

        def _init_weights(self):
            targets = [self.classifier, self.vad_head, self.kpi_head, self.signal_heads]
            if self.use_wavlm:
                targets.append(self.backbone.proj)
            else:
                targets.extend([self.prosody_embed, self.phonetic_embed, self.fusion])
            if self.use_dual_stream_fusion and self.use_wavlm:
                targets.extend([self.prosody_stream, self.stream_gate])
            if self.n_speakers > 0:
                targets.append(self.speaker_head)
            if not MAMBA_SSM_AVAILABLE:
                targets.extend(list(self.blocks))
            for target in targets:
                for module in target.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

        def _specaugment(self, x: torch.Tensor) -> torch.Tensor:
            """Apply SpecAugment-style masking to sequence features during training."""
            if not self.training or self.specaugment_prob <= 0:
                return x
            batch, seq_len, d = x.shape
            for i in range(batch):
                if torch.rand(1).item() > self.specaugment_prob:
                    continue
                # Time masking
                if self.specaugment_time_mask > 0 and seq_len > 1:
                    t = min(int(torch.randint(1, self.specaugment_time_mask + 1, (1,)).item()), seq_len)
                    t0 = int(torch.randint(0, max(seq_len - t, 1), (1,)).item())
                    x[i, t0:t0 + t, :] = 0.0
                # Feature/frequency masking
                if self.specaugment_freq_mask > 0 and d > 1:
                    f = min(int(torch.randint(1, self.specaugment_freq_mask + 1, (1,)).item()), d)
                    f0 = int(torch.randint(0, max(d - f, 1), (1,)).item())
                    x[i, :, f0:f0 + f] = 0.0
            return x

        def pool(self, sequence: torch.Tensor) -> torch.Tensor:
            """Mean-pool a sequence (batch, seq_len, d_model) -> (batch, d_model)."""
            return sequence.mean(dim=1)

        def encode(
            self,
            waveform: Optional[torch.Tensor] = None,
            prosody_features: Optional[torch.Tensor] = None,
            phonetic_features: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Encode input into sequence and pooled representations.

            Returns:
                (sequence_repr, pooled_repr)
                sequence_repr: (batch, seq_len, d_model)
                pooled_repr: (batch, d_model)
            """
            if self.use_wavlm:
                x = self.backbone(waveform)
                if self.use_dual_stream_fusion and prosody_features is not None:
                    # Align prosody stream length to WavLM output
                    p = self.prosody_stream(prosody_features)
                    if p.shape[1] != x.shape[1]:
                        p = F.interpolate(
                            p.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False
                        ).transpose(1, 2)
                    gate = self.stream_gate(torch.cat([x, p], dim=-1))
                    x = gate * x + (1 - gate) * p
            else:
                batch, seq_len, _ = prosody_features.shape
                prosody_emb = self.prosody_embed(prosody_features)

                if phonetic_features is not None:
                    phonetic_emb = self.phonetic_embed(phonetic_features)
                else:
                    phonetic_emb = torch.zeros(
                        batch, seq_len, prosody_emb.shape[-1],
                        device=prosody_features.device, dtype=prosody_features.dtype,
                    )

                x = torch.cat([prosody_emb, phonetic_emb], dim=-1)
                x = self.fusion(x)

            x = self._specaugment(x)

            for block in self.blocks:
                x = block(x)

            return x, self.pool(x)

        def forward(
            self,
            waveform: Optional[torch.Tensor] = None,
            prosody_features: Optional[torch.Tensor] = None,
            phonetic_features: Optional[torch.Tensor] = None,
            kpi_type: Optional[torch.Tensor] = None,
            kpi_direction: Optional[torch.Tensor] = None,
            kpi_range: Optional[torch.Tensor] = None,
        ) -> dict[str, torch.Tensor]:
            """
            Forward pass: emotion classification, VAD, and KPI prediction.

            Returns:
                Dict with:
                    'emotion_logits': (batch, n_emotions)
                    'vad': (batch, 3) — valence, arousal, dominance
                    'repr': (batch, d_model) — pooled representation
                    'sequence_repr': (batch, seq_len, d_model) — full sequence
                    'speaker_logits': (batch, n_speakers) — if speaker head enabled
                    'kpi_value': (batch,) — if kpi_type provided
                    'kpi_category_logits': (batch, max_categories) — for categorical KPIs
            """
            sequence_repr, repr = self.encode(
                waveform=waveform,
                prosody_features=prosody_features,
                phonetic_features=phonetic_features,
            )

            emotion_logits = self.classifier(repr)
            vad = self.vad_head(repr)

            signal_values = self.signal_heads(repr)
            signals = dict(zip(self.SIGNAL_NAMES, signal_values.unbind(-1), strict=True))

            result = {
                "emotion_logits": emotion_logits,
                "vad": vad,
                "repr": repr,
                "sequence_repr": sequence_repr,
                "signals": signals,
            }

            if self.n_speakers > 0 and hasattr(self, "speaker_head"):
                grl_repr = _GradientReversalFn.apply(repr, self.speaker_grl_alpha)
                result["speaker_logits"] = self.speaker_head(grl_repr)

            if kpi_type is not None:
                kpi_out = self.kpi_head(
                    prosody_repr=repr,
                    kpi_type=kpi_type,
                    kpi_direction=kpi_direction,
                    kpi_range=kpi_range,
                )
                result["kpi_value"] = kpi_out["value"]
                result["kpi_category_logits"] = kpi_out["category_logits"]

            return result

        @torch.no_grad()
        def step(
            self,
            waveform_chunk: Optional[torch.Tensor] = None,
            prosody_frame: Optional[torch.Tensor] = None,
            phonetic_frame: Optional[torch.Tensor] = None,
            state: Optional[dict] = None,
        ) -> tuple[torch.Tensor, torch.Tensor, dict]:
            """
            Streaming forward for real-time prosodic signal generation.

        WavLM + Mamba path (use_wavlm=True):
                waveform_chunk: (batch, samples) — raw audio chunk at 16kHz.
                Typical chunk size: 0.5-2s of audio. WavLM processes the chunk,
                then Mamba steps through each WavLM frame using inference_params.
                The Mamba state persists across calls, accumulating temporal context.

            Legacy MFCC path (use_wavlm=False, no mamba_ssm):
                prosody_frame: (batch, prosody_dim) — single MFCC frame.
                S4D step-by-step recurrence.

            Returns:
                emotion_probs: (batch, n_emotions)
                vad: (batch, 3) — valence, arousal, dominance
                new_state: updated state dict (includes 'repr' for KPI head)
            """
            self.eval()

            if state is None:
                state = {}

            if self.use_wavlm:
                if waveform_chunk is None:
                    raise ValueError("waveform_chunk required when use_wavlm=True")

                batch = waveform_chunk.shape[0]
                x = self.backbone(waveform_chunk)  # (batch, seq_len, d_model)

                if MAMBA_SSM_AVAILABLE:
                    from mamba_ssm.utils.generation import InferenceParams

                    inference_params = state.get('inference_params', None)
                    if inference_params is None:
                        inference_params = InferenceParams(
                            max_seqlen=1,
                            max_batch_size=batch,
                        )
                        for block in self.blocks:
                            block.mamba.allocate_inference_cache(
                                batch, max_seqlen=1,
                            )
                    seqlen_offset = state.get('seqlen_offset', 0)

                    for t in range(x.shape[1]):
                        frame = x[:, t:t+1, :]
                        inference_params.seqlen_offset = seqlen_offset + t
                        for block in self.blocks:
                            residual = frame
                            frame = block.norm(frame)
                            frame = block.mamba(frame, inference_params=inference_params)
                            frame = block.dropout(frame)
                            frame = frame + residual

                    x_out = frame.squeeze(1)  # (batch, d_model)
                    new_state = {
                        'inference_params': inference_params,
                        'seqlen_offset': seqlen_offset + x.shape[1],
                    }
                else:
                    block_states = state.get('blocks', [None] * len(self.blocks))
                    new_block_states = []
                    for t in range(x.shape[1]):
                        frame = x[:, t, :]
                        for i, block in enumerate(self.blocks):
                            frame, blk_state = block.step(frame, block_states[i])
                            if t == x.shape[1] - 1:
                                new_block_states.append(blk_state)
                    x_out = frame
                    new_state = {'blocks': new_block_states}

            else:
                if prosody_frame is None:
                    raise ValueError("prosody_frame required when use_wavlm=False")

                batch = prosody_frame.shape[0]
                device = prosody_frame.device

                if phonetic_frame is None:
                    phonetic_frame = torch.zeros(batch, self.phonetic_dim, device=device)

                prosody_emb = self.prosody_embed(prosody_frame)
                phonetic_emb = self.phonetic_embed(phonetic_frame)
                x = torch.cat([prosody_emb, phonetic_emb], dim=-1)
                x = self.fusion(x)

                block_states = state.get('blocks', [None] * len(self.blocks))
                new_block_states = []
                for i, block in enumerate(self.blocks):
                    x, blk_state = block.step(x, block_states[i])
                    new_block_states.append(blk_state)

                x_out = x
                new_state = {'blocks': new_block_states}

            emotion_logits = self.classifier(x_out)
            emotion_probs = F.softmax(emotion_logits, dim=-1)
            vad = self.vad_head(x_out)
            signal_values = self.signal_heads(x_out)
            signals = dict(zip(self.SIGNAL_NAMES, signal_values.unbind(-1), strict=True))

            new_state['repr'] = x_out
            new_state['signals'] = signals
            return emotion_probs, vad, new_state

        def predict(
            self,
            prosody_features: ProsodyFeatures,
            phonetic_features: PhoneticFeatures,
        ) -> EmotionPrediction:
            """
            Emotion prediction from the pretraining head (MFCC path only).

            For WavLM-based inference, use forward(waveform=...) directly.
            """
            if self.use_wavlm:
                raise ValueError(
                    "predict() uses MFCC features, incompatible with use_wavlm=True. "
                    "Use forward(waveform=...) for WavLM-based inference."
                )

            self.eval()

            prosody_vec = torch.tensor(
                prosody_features.to_vector(),
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0)

            phonetic_vec = torch.tensor(
                phonetic_features.to_vector(),
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                result = self.forward(
                    prosody_features=prosody_vec,
                    phonetic_features=phonetic_vec,
                )
                probs = F.softmax(result["emotion_logits"], dim=-1)[0]

            emotion_idx = probs.argmax().item()
            emotion_labels = list(EmotionLabel)
            primary_emotion = emotion_labels[emotion_idx]

            emotion_probs = {
                label.value: float(probs[i])
                for i, label in enumerate(emotion_labels)
            }

            vad_values = result["vad"][0].cpu().numpy()

            return EmotionPrediction(
                primary_emotion=primary_emotion,
                confidence=float(probs[emotion_idx]),
                emotion_probabilities=emotion_probs,
                valence=float(vad_values[0]),
                arousal=float((vad_values[1] + 1) / 2),
                dominance=float((vad_values[2] + 1) / 2),
            )

        def predict_prosody(
            self,
            prosody_features: ProsodyFeatures,
            phonetic_features: Optional[PhoneticFeatures] = None,
        ) -> ProsodyPrediction:
            """
            Prosodic state prediction (MFCC path only).

            For WavLM-based inference, use forward(waveform=...) directly.
            """
            if self.use_wavlm:
                raise ValueError(
                    "predict_prosody() uses MFCC features, incompatible with use_wavlm=True. "
                    "Use forward(waveform=...) for WavLM-based inference."
                )

            self.eval()

            prosody_vec = torch.tensor(
                prosody_features.to_vector(),
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0)

            phonetic_vec = None
            if phonetic_features is not None:
                phonetic_vec = torch.tensor(
                    phonetic_features.to_vector(),
                    dtype=torch.float32,
                ).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                result = self.forward(
                    prosody_features=prosody_vec,
                    phonetic_features=phonetic_vec,
                )

            vad = result["vad"][0].cpu().numpy()

            return ProsodyPrediction(
                valence=float(vad[0]),
                arousal=float((vad[1] + 1) / 2),   # scale from [-1,1] to [0,1]
                dominance=float((vad[2] + 1) / 2),
                confidence=0.8,  # placeholder; real confidence from ensemble/dropout
            )

        def predict_kpi(
            self,
            prosody_features: ProsodyFeatures,
            kpi_config: KPIModelConfig,
            phonetic_features: Optional[PhoneticFeatures] = None,
        ) -> KPIModelOutput:
            """
            Predict a single KPI outcome from prosodic features (MFCC path only).

            For WavLM-based inference, use forward(waveform=...) directly
            and pass the representation to self.kpi_head.
            """
            if self.use_wavlm:
                raise ValueError(
                    "predict_kpi() uses MFCC features, incompatible with use_wavlm=True. "
                    "Use forward(waveform=...) for WavLM-based inference."
                )

            self.eval()

            prosody_vec = torch.tensor(
                prosody_features.to_vector(),
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0)

            phonetic_vec = None
            if phonetic_features is not None:
                phonetic_vec = torch.tensor(
                    phonetic_features.to_vector(),
                    dtype=torch.float32,
                ).unsqueeze(0).unsqueeze(0)

            kpi_type_t = torch.tensor([kpi_config.kpi_type], dtype=torch.long)
            kpi_dir_t = torch.tensor([kpi_config.direction], dtype=torch.long)
            kpi_range_t = torch.tensor(
                [[kpi_config.range_min, kpi_config.range_max]],
                dtype=torch.float32,
            )

            with torch.no_grad():
                result = self.forward(
                    prosody_features=prosody_vec,
                    phonetic_features=phonetic_vec,
                    kpi_type=kpi_type_t,
                    kpi_direction=kpi_dir_t,
                    kpi_range=kpi_range_t,
                )

            raw_value = float(result["kpi_value"][0])

            if kpi_config.kpi_type == 1:  # binary
                value = float(torch.sigmoid(torch.tensor(raw_value)))
            elif kpi_config.kpi_type == 0:  # scalar
                value = max(kpi_config.range_min, min(kpi_config.range_max, raw_value))
            else:  # categorical
                value = raw_value

            category_logits = None
            if kpi_config.kpi_type == 2 and "kpi_category_logits" in result:
                logits = result["kpi_category_logits"][0][:kpi_config.n_categories]
                category_logits = logits.cpu().tolist()

            return KPIModelOutput(
                value=value,
                confidence=0.8,  # placeholder
                category_logits=category_logits,
            )

        _VALID_CONFIG_KEYS = {
            "use_wavlm", "prosody_dim", "phonetic_dim", "d_model", "n_layers",
            "d_state", "n_emotions", "d_kpi_embed", "max_categories", "dropout",
            "use_weighted_wavlm_layers", "use_dual_stream_fusion",
            "n_speakers", "speaker_grl_alpha",
            "specaugment_prob", "specaugment_time_mask", "specaugment_freq_mask",
        }

        @classmethod
        def from_pretrained(cls, model_path: str) -> "ProsodySSM":
            """Load a pretrained model from a .pt file or a HuggingFace-style directory (config.json + pytorch_model.bin or prosody_model.pt)."""
            import json
            from pathlib import Path
            path = Path(model_path)
            if path.is_dir():
                config_path = path / "config.json"
                if config_path.exists():
                    config = json.loads(config_path.read_text())
                else:
                    config = {}
                filtered_config = {k: v for k, v in config.items() if k in cls._VALID_CONFIG_KEYS}
                if "use_wavlm" not in filtered_config:
                    filtered_config["use_wavlm"] = config.get("use_wavlm", True)
                if "n_layers" not in filtered_config and "n_layers" in config:
                    filtered_config["n_layers"] = config["n_layers"]
                model = cls(**filtered_config)
                weights_path = path / "pytorch_model.bin"
                if weights_path.exists():
                    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=False)
                else:
                    pt_path = path / "prosody_model.pt"
                    if pt_path.exists():
                        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                        model.load_state_dict(ckpt["model_state_dict"], strict=False)
                    else:
                        raise FileNotFoundError(f"No pytorch_model.bin or prosody_model.pt in {path}")
                return model
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            config = checkpoint.get("config", {})
            filtered_config = {k: v for k, v in config.items() if k in cls._VALID_CONFIG_KEYS}
            if "use_wavlm" not in filtered_config:
                filtered_config["use_wavlm"] = False
            model = cls(**filtered_config)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            return model

        def save_pretrained(self, model_path: str, config: Optional[dict] = None):
            """Save model to path."""
            if config is None:
                config = {}
            full_config = {
                'use_wavlm': config.get('use_wavlm', self.use_wavlm),
                'd_model': config.get('d_model', self.d_model),
                'n_layers': config.get('n_layers', len(self.blocks)),
                'd_state': config.get('d_state', self.d_state),
                'n_emotions': config.get('n_emotions', self.n_emotions),
                'dropout': config.get('dropout', self.blocks[0].dropout.p),
                'use_weighted_wavlm_layers': self.use_weighted_wavlm_layers,
                'use_dual_stream_fusion': self.use_dual_stream_fusion,
                'n_speakers': self.n_speakers,
                'speaker_grl_alpha': self.speaker_grl_alpha,
                'specaugment_prob': self.specaugment_prob,
                'specaugment_time_mask': self.specaugment_time_mask,
                'specaugment_freq_mask': self.specaugment_freq_mask,
            }
            if not self.use_wavlm:
                full_config['prosody_dim'] = config.get('prosody_dim', self.prosody_dim)
                full_config['phonetic_dim'] = config.get('phonetic_dim', self.phonetic_dim)
            from pathlib import Path
            dest = Path(model_path)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': full_config,
            }, tmp)
            tmp.rename(dest)


    # -------------------------------------------------------------------
    # Training loss
    # -------------------------------------------------------------------

    def compute_kpi_loss(
        predictions: dict[str, torch.Tensor],
        kpi_type: torch.Tensor,
        targets: torch.Tensor,
        kpi_range: Optional[torch.Tensor] = None,
        n_categories: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss for KPI-conditioned predictions.

        Handles all KPI types:
            - scalar: MSE (normalized to KPI range)
            - binary: BCE with logits
            - categorical: Cross-entropy

        Args:
            predictions: dict from ProsodySSM.forward()
            kpi_type: (batch,) int — 0=scalar, 1=binary, 2=categorical
            targets: (batch,) float — actual KPI values
                For scalar: the value itself
                For binary: 0.0 or 1.0
                For categorical: class index (as float, will be cast to long)
            kpi_range: (batch, 2) — for normalizing scalar loss
            n_categories: (batch,) int — for masking category logits

        Returns:
            Scalar loss tensor
        """
        kpi_value = predictions["kpi_value"]  # (batch,) -- raw predicted values
        kpi_type.shape[0]
        device = kpi_type.device

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        # --- Scalar KPIs (e.g., CSAT 1-5, NPS -100 to 100, handle_time 0-3600) ---
        # MSE loss, but normalized to [0,1] by KPI range so that a CSAT error of 1
        # weighs the same as an NPS error of 20. Without normalization, wide-range
        # KPIs would dominate the loss.
        scalar_mask = kpi_type == 0
        if scalar_mask.any():
            pred_s = kpi_value[scalar_mask]
            tgt_s = targets[scalar_mask]

            if kpi_range is not None:
                range_s = kpi_range[scalar_mask]
                span = (range_s[:, 1] - range_s[:, 0]).clamp(min=1e-6)
                pred_norm = (pred_s - range_s[:, 0]) / span
                tgt_norm = (tgt_s - range_s[:, 0]) / span
                total_loss = total_loss + F.mse_loss(pred_norm, tgt_norm)
            else:
                total_loss = total_loss + F.mse_loss(pred_s, tgt_s)
            count += 1

        # --- Binary KPIs (e.g., churned, escalated, deal_won) ---
        # BCE with logits -- the model outputs raw logits, sigmoid is applied
        # inside the loss for numerical stability.
        binary_mask = kpi_type == 1
        if binary_mask.any():
            pred_b = kpi_value[binary_mask]
            tgt_b = targets[binary_mask]
            total_loss = total_loss + F.binary_cross_entropy_with_logits(pred_b, tgt_b)
            count += 1

        # --- Categorical KPIs (e.g., call disposition, sentiment category) ---
        # Standard cross-entropy over category logits.
        cat_mask = kpi_type == 2
        if cat_mask.any() and "kpi_category_logits" in predictions:
            logits_c = predictions["kpi_category_logits"][cat_mask]
            tgt_c = targets[cat_mask].long()
            total_loss = total_loss + F.cross_entropy(logits_c, tgt_c)
            count += 1

        # Average across active KPI types in this batch.
        # A batch can mix scalar + binary + categorical -- each contributes equally.
        return total_loss / max(count, 1)


    # -------------------------------------------------------------------
    # Backward compatibility alias
    # -------------------------------------------------------------------

    ProsodySSMClassifier = ProsodySSM


else:
    # ---------------------------------------------------------------
    # Fallback when PyTorch is not available
    # ---------------------------------------------------------------

    class ProsodySSM:
        """Heuristic fallback when PyTorch is not available."""

        def __init__(self, **kwargs):
            self._warned = False

        def _warn_once(self):
            if not self._warned:
                import warnings
                warnings.warn(
                    "PyTorch not available. Using heuristic predictions. "
                    "Install PyTorch for the full SSM model.", stacklevel=2
                )
                self._warned = True

        def _heuristic_vad(self, prosody_features: ProsodyFeatures) -> tuple[float, float, float]:
            """Compute rough VAD from prosodic features."""
            f0_mean = prosody_features.f0_mean
            energy_mean = prosody_features.energy_mean
            speech_rate = prosody_features.speech_rate

            arousal = min(1.0, (energy_mean * 5 + speech_rate / 5) / 2)

            valence = 0.0
            if f0_mean > 200 and prosody_features.f0_range > 100:
                valence = 0.3
            elif f0_mean < 130 and energy_mean < 0.03:
                valence = -0.3

            dominance = min(1.0, max(0.0, 0.3 + energy_mean * 3))

            return valence, arousal, dominance

        def predict(
            self,
            prosody_features: ProsodyFeatures,
            phonetic_features: PhoneticFeatures,
        ) -> EmotionPrediction:
            """Legacy heuristic emotion prediction."""
            self._warn_once()

            f0_mean = prosody_features.f0_mean
            f0_range = prosody_features.f0_range
            energy_mean = prosody_features.energy_mean
            speech_rate = prosody_features.speech_rate

            scores = dict.fromkeys(EmotionLabel, 0.0)

            if f0_mean > 200 and energy_mean > 0.1 and speech_rate > 3:
                if f0_range > 100:
                    scores[EmotionLabel.HAPPY] += 0.4
                else:
                    scores[EmotionLabel.ANGRY] += 0.4

            if f0_mean < 150 and energy_mean < 0.05:
                if speech_rate < 2:
                    scores[EmotionLabel.SAD] += 0.4
                else:
                    scores[EmotionLabel.NEUTRAL] += 0.3

            if f0_range > 150:
                scores[EmotionLabel.SURPRISED] += 0.3

            scores[EmotionLabel.NEUTRAL] += 0.2

            total = sum(scores.values())
            probs = {k.value: v / total for k, v in scores.items()}
            primary = max(scores, key=scores.get)

            valence, arousal, dominance = self._heuristic_vad(prosody_features)

            return EmotionPrediction(
                primary_emotion=primary,
                confidence=scores[primary] / total,
                emotion_probabilities=probs,
                valence=valence,
                arousal=arousal,
                dominance=dominance,
            )

        def predict_prosody(
            self,
            prosody_features: ProsodyFeatures,
            phonetic_features: Optional[PhoneticFeatures] = None,
        ) -> ProsodyPrediction:
            """Heuristic VAD estimation."""
            self._warn_once()
            valence, arousal, dominance = self._heuristic_vad(prosody_features)
            return ProsodyPrediction(
                valence=valence,
                arousal=arousal,
                dominance=dominance,
                confidence=0.4,
            )

        def predict_kpi(
            self,
            prosody_features: ProsodyFeatures,
            kpi_config: KPIModelConfig,
            phonetic_features: Optional[PhoneticFeatures] = None,
        ) -> KPIModelOutput:
            """Heuristic KPI prediction (placeholder)."""
            pred = self.predict_prosody(prosody_features, phonetic_features)

            if kpi_config.kpi_type == 0:  # scalar
                midpoint = (kpi_config.range_min + kpi_config.range_max) / 2
                half = (kpi_config.range_max - kpi_config.range_min) / 2
                value = midpoint + pred.valence * half * 0.5
                value = max(kpi_config.range_min, min(kpi_config.range_max, value))
            elif kpi_config.kpi_type == 1:  # binary
                value = 0.5 + pred.valence * 0.2
                value = max(0.0, min(1.0, value))
            else:
                value = 0.0

            return KPIModelOutput(value=value, confidence=0.3)

        def eval(self):
            pass

        @classmethod
        def from_pretrained(cls, model_path: str) -> "ProsodySSM":
            return cls()

        def save_pretrained(self, model_path: str, config=None):
            pass

    ProsodySSMClassifier = ProsodySSM
