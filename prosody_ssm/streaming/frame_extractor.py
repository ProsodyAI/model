"""
Frame-level prosodic feature extraction for streaming audio.

Processes 50ms PCM frames and maintains running statistics for
pitch, energy, voice quality, rhythm, and spectral features.
No ASR dependency -- pure prosodic signal.

Output: 28-dim prosody vector per frame, matching ProsodySSM input dimension.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class ProsodyFrame:
    """Single frame of prosodic features (50ms window)."""

    # Pitch (F0)
    f0: float = 0.0               # Instantaneous F0 (Hz), 0 if unvoiced
    voiced: bool = False           # Whether frame is voiced

    # Energy
    energy: float = 0.0            # RMS energy
    energy_db: float = -80.0       # Energy in dB

    # Spectral
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    mfccs: np.ndarray = field(default_factory=lambda: np.zeros(13))

    # Running statistics (updated across frames)
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_range: float = 0.0
    energy_mean: float = 0.0
    energy_std: float = 0.0

    # Voice quality (estimated from short-term signal)
    jitter: float = 0.0
    shimmer: float = 0.0
    hnr: float = 0.0

    # Rhythm (from pause/voice patterns)
    speech_rate: float = 0.0       # Estimated from voicing patterns
    pause_rate: float = 0.0

    # Timestamp
    frame_index: int = 0
    timestamp_ms: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to 28-dim vector matching ProsodySSM input."""
        return np.array([
            # Pitch features (5)
            self.f0_mean, self.f0_std, self.f0 if self.voiced else self.f0_mean,
            self.f0_mean + self.f0_range / 2, self.f0_range,
            # Energy features (2)
            self.energy_mean, self.energy_std,
            # Voice quality (3)
            self.jitter, self.shimmer, self.hnr,
            # Rhythm (3)
            self.speech_rate, self.pause_rate, 0.0,  # pause_duration placeholder
            # Spectral (2)
            self.spectral_centroid, self.spectral_rolloff,
            # MFCCs (13)
            *self.mfccs[:13],
        ], dtype=np.float32)


class FrameExtractor:
    """
    Streaming prosodic feature extractor.

    Processes 50ms PCM frames (800 samples at 16kHz) and maintains
    running statistics across the conversation. Outputs a 28-dim
    prosody vector per frame matching ProsodySSM's input dimension.

    Usage:
        extractor = FrameExtractor(sample_rate=16000)
        for pcm_chunk in audio_stream:
            frame = extractor.process_frame(pcm_chunk)
            # frame.to_vector() -> np.ndarray(28,)
    """

    # Frame config
    FRAME_MS = 50           # Frame duration in ms
    HOP_MS = 25             # Hop between frames
    F0_MIN = 50.0           # Minimum F0 (Hz)
    F0_MAX = 500.0          # Maximum F0 (Hz)
    N_MFCC = 13             # Number of MFCCs

    # Running stat window (last N frames for statistics)
    STAT_WINDOW = 40        # ~2 seconds of history at 50ms frames

    def __init__(self, sample_rate: int = 16000):
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for frame extraction")

        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * self.FRAME_MS / 1000)  # 800 samples
        self.hop_size = int(sample_rate * self.HOP_MS / 1000)      # 400 samples

        # Running statistics
        self._f0_history: deque[float] = deque(maxlen=self.STAT_WINDOW)
        self._energy_history: deque[float] = deque(maxlen=self.STAT_WINDOW)
        self._voiced_history: deque[bool] = deque(maxlen=self.STAT_WINDOW)
        self._period_history: deque[float] = deque(maxlen=10)  # For jitter
        self._amplitude_history: deque[float] = deque(maxlen=10)  # For shimmer

        # Frame counter
        self._frame_index = 0

        # Input buffer for accumulating partial frames
        self._buffer = np.array([], dtype=np.float32)

    def process_frame(self, pcm: bytes | np.ndarray) -> Optional[ProsodyFrame]:
        """
        Process a chunk of PCM audio and return a ProsodyFrame.

        Args:
            pcm: Raw PCM audio as bytes (int16) or float32 numpy array.
                 Can be any length; internally buffered to 50ms frames.

        Returns:
            ProsodyFrame if enough audio accumulated, None otherwise.
        """
        # Convert bytes to float32
        if isinstance(pcm, bytes):
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = pcm.astype(np.float32)

        # Accumulate in buffer
        self._buffer = np.concatenate([self._buffer, audio])

        # Need at least one full frame
        if len(self._buffer) < self.frame_size:
            return None

        # Extract one frame from the buffer
        frame_audio = self._buffer[:self.frame_size]
        self._buffer = self._buffer[self.hop_size:]  # Advance by hop

        return self._extract(frame_audio)

    def process_frames(self, pcm: bytes | np.ndarray) -> list[ProsodyFrame]:
        """Process audio and return all complete frames."""
        if isinstance(pcm, bytes):
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = pcm.astype(np.float32)

        self._buffer = np.concatenate([self._buffer, audio])

        frames = []
        while len(self._buffer) >= self.frame_size:
            frame_audio = self._buffer[:self.frame_size]
            self._buffer = self._buffer[self.hop_size:]
            frames.append(self._extract(frame_audio))

        return frames

    def _extract(self, audio: np.ndarray) -> ProsodyFrame:
        """Extract features from a single frame of audio."""
        frame = ProsodyFrame()
        frame.frame_index = self._frame_index
        frame.timestamp_ms = self._frame_index * self.HOP_MS
        self._frame_index += 1

        # --- Pitch (F0) ---
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=self.F0_MIN,
            fmax=self.F0_MAX,
            sr=self.sample_rate,
            frame_length=len(audio),
            hop_length=len(audio),
        )

        if f0 is not None and len(f0) > 0 and not np.isnan(f0[0]):
            frame.f0 = float(f0[0])
            frame.voiced = bool(voiced_flag[0]) if voiced_flag is not None else True
        else:
            frame.f0 = 0.0
            frame.voiced = False

        # Update F0 running stats
        if frame.voiced and frame.f0 > 0:
            self._f0_history.append(frame.f0)
            period = 1.0 / frame.f0
            self._period_history.append(period)
        self._voiced_history.append(frame.voiced)

        if self._f0_history:
            f0_arr = np.array(list(self._f0_history))
            frame.f0_mean = float(np.mean(f0_arr))
            frame.f0_std = float(np.std(f0_arr))
            frame.f0_range = float(np.max(f0_arr) - np.min(f0_arr))

        # --- Energy ---
        rms = float(np.sqrt(np.mean(audio ** 2)))
        frame.energy = rms
        frame.energy_db = float(20 * np.log10(rms + 1e-10))
        self._energy_history.append(rms)

        if self._energy_history:
            e_arr = np.array(list(self._energy_history))
            frame.energy_mean = float(np.mean(e_arr))
            frame.energy_std = float(np.std(e_arr))

        # Amplitude for shimmer
        self._amplitude_history.append(rms)

        # --- Voice quality ---
        # Jitter: variation in pitch periods
        if len(self._period_history) >= 3:
            periods = np.array(list(self._period_history))
            diffs = np.abs(np.diff(periods))
            frame.jitter = float(np.mean(diffs) / (np.mean(periods) + 1e-10))

        # Shimmer: variation in amplitude
        if len(self._amplitude_history) >= 3:
            amps = np.array(list(self._amplitude_history))
            diffs = np.abs(np.diff(amps))
            frame.shimmer = float(np.mean(diffs) / (np.mean(amps) + 1e-10))

        # HNR estimate from autocorrelation
        if frame.voiced and len(audio) > 0:
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            if len(autocorr) > 1 and autocorr[0] > 0:
                r1 = autocorr[1] / autocorr[0]
                frame.hnr = float(10 * np.log10(max(r1, 1e-10) / max(1 - r1, 1e-10)))

        # --- Spectral ---
        frame.spectral_centroid = float(librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=len(audio)
        ).mean())

        frame.spectral_rolloff = float(librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=len(audio)
        ).mean())

        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=self.N_MFCC,
            hop_length=len(audio),
        )
        frame.mfccs = mfccs.flatten()[:self.N_MFCC]

        # --- Rhythm ---
        # Speech rate estimated from voicing pattern
        if len(self._voiced_history) >= 10:
            voiced_arr = np.array(list(self._voiced_history))
            # Count voiced-to-unvoiced transitions as syllable boundaries
            transitions = np.sum(np.abs(np.diff(voiced_arr.astype(float))))
            duration_sec = len(voiced_arr) * self.FRAME_MS / 1000
            frame.speech_rate = float(transitions / 2 / duration_sec) if duration_sec > 0 else 0.0

            # Pause rate
            unvoiced = np.sum(~voiced_arr)
            frame.pause_rate = float(unvoiced / len(voiced_arr))

        return frame

    def reset(self):
        """Reset extractor state for a new conversation."""
        self._f0_history.clear()
        self._energy_history.clear()
        self._voiced_history.clear()
        self._period_history.clear()
        self._amplitude_history.clear()
        self._frame_index = 0
        self._buffer = np.array([], dtype=np.float32)
