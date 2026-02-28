"""
PyTorch Dataset classes for speech emotion recognition.

Supports:
- Local file loading
- S3 streaming for cloud training
- On-the-fly feature extraction
- Caching for faster training
"""

import io
import json
import os
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import boto3
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from google.cloud import storage as gcs_storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


# Import canonical emotion labels from model
from prosody_ssm.model import EmotionLabel

EMOTION_LABELS = [e.value for e in EmotionLabel]


class EmotionSpeechDataset(Dataset):
    """
    PyTorch Dataset for speech emotion recognition.
    
    Loads audio files and extracts features on-the-fly or from cache.
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Union[str, Path],
        sample_rate: int = 16000,
        max_length_sec: float = 10.0,
        feature_extractor: Optional[Callable] = None,
        cache_features: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            manifest_path: Path to JSON manifest file
            data_root: Root directory for audio files
            sample_rate: Target sample rate
            max_length_sec: Maximum audio length in seconds
            feature_extractor: Optional feature extraction function
            cache_features: Whether to cache extracted features
            cache_dir: Directory for cached features
            transform: Optional transform to apply to features
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for dataset classes")
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio loading")
        
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.max_length = int(max_length_sec * sample_rate)
        self.feature_extractor = feature_extractor
        self.cache_features = cache_features
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform
        
        # Load manifest
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        
        # Setup cache
        if self.cache_features and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.manifest[idx]
        audio_path = self.data_root / item["audio_path"]
        
        # Check cache first
        if self.cache_features and self.cache_dir:
            cache_path = self.cache_dir / f"{idx}.npz"
            if cache_path.exists():
                cached = np.load(cache_path)
                features = {k: torch.from_numpy(v) for k, v in cached.items()}
                features["emotion_idx"] = torch.tensor(item["emotion_idx"], dtype=torch.long)
                return features
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Pad or truncate
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        
        # Extract features or return raw audio
        if self.feature_extractor:
            features = self.feature_extractor(audio, sr)
            
            # Cache if enabled
            if self.cache_features and self.cache_dir:
                cache_path = self.cache_dir / f"{idx}.npz"
                np.savez(cache_path, **{k: v.numpy() if hasattr(v, 'numpy') else v for k, v in features.items()})
        else:
            features = {"audio": torch.from_numpy(audio).float()}
        
        # Add label
        features["emotion_idx"] = torch.tensor(item["emotion_idx"], dtype=torch.long)
        
        # Apply transform
        if self.transform:
            features = self.transform(features)
        
        return features
    
    @property
    def num_classes(self) -> int:
        return len(EMOTION_LABELS)


class S3EmotionSpeechDataset(Dataset):
    """
    PyTorch Dataset that streams audio from S3.
    
    Efficiently loads audio data from S3 bucket for cloud training.
    """
    
    def __init__(
        self,
        manifest_path: str,  # Can be S3 URI or local path
        s3_bucket: str,
        s3_prefix: str = "",
        sample_rate: int = 16000,
        max_length_sec: float = 10.0,
        feature_extractor: Optional[Callable] = None,
        aws_region: str = "us-east-1",
        transform: Optional[Callable] = None,
        local_cache_dir: Optional[str] = None,
    ):
        """
        Initialize S3 dataset.
        
        Args:
            manifest_path: Path to manifest (local or s3://bucket/path)
            s3_bucket: S3 bucket name
            s3_prefix: Prefix for audio files in bucket
            sample_rate: Target sample rate
            max_length_sec: Maximum audio length
            feature_extractor: Optional feature extraction function
            aws_region: AWS region
            transform: Optional transform
            local_cache_dir: Local directory to cache downloaded files
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 access. Install with: pip install boto3")
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio loading")
        
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.strip("/")
        self.sample_rate = sample_rate
        self.max_length = int(max_length_sec * sample_rate)
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
        
        # Initialize S3 client with retry config
        config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=50,
        )
        self.s3_client = boto3.client("s3", region_name=aws_region, config=config)
        
        # Load manifest
        self.manifest = self._load_manifest(manifest_path)
        
        # Setup local cache
        if self.local_cache_dir:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_manifest(self, manifest_path: str) -> list[dict]:
        """Load manifest from local file or S3."""
        if manifest_path.startswith("s3://"):
            # Parse S3 URI
            parts = manifest_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            manifest = json.loads(response["Body"].read().decode("utf-8"))
        else:
            with open(manifest_path) as f:
                manifest = json.load(f)
        
        return manifest
    
    def _download_from_s3(self, s3_key: str) -> bytes:
        """Download file from S3."""
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
        return response["Body"].read()
    
    def _get_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """Get audio from S3 or local cache."""
        s3_key = f"{self.s3_prefix}/{audio_path}" if self.s3_prefix else audio_path
        
        # Check local cache
        if self.local_cache_dir:
            local_path = self.local_cache_dir / audio_path.replace("/", "_")
            if local_path.exists():
                return librosa.load(local_path, sr=self.sample_rate)
        
        # Download from S3
        audio_bytes = self._download_from_s3(s3_key)
        
        # Save to local cache
        if self.local_cache_dir:
            local_path = self.local_cache_dir / audio_path.replace("/", "_")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(audio_bytes)
        
        # Load audio from bytes
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate)
        return audio, sr
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.manifest[idx]
        
        # Load audio from S3
        audio, sr = self._get_audio(item["audio_path"])
        
        # Pad or truncate
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        
        # Extract features
        if self.feature_extractor:
            features = self.feature_extractor(audio, sr)
        else:
            features = {"audio": torch.from_numpy(audio).float()}
        
        # Add label
        features["emotion_idx"] = torch.tensor(item["emotion_idx"], dtype=torch.long)
        
        # Apply transform
        if self.transform:
            features = self.transform(features)
        
        return features
    
    @property
    def num_classes(self) -> int:
        return len(EMOTION_LABELS)


class GCSEmotionSpeechDataset(Dataset):
    """
    PyTorch Dataset that streams audio from Google Cloud Storage.
    
    Efficiently loads audio data from GCS bucket for cloud training on GCP.
    """
    
    def __init__(
        self,
        manifest_path: str,  # Can be gs:// URI or local path
        gcs_bucket: str,
        gcs_prefix: str = "",
        sample_rate: int = 16000,
        max_length_sec: float = 10.0,
        feature_extractor: Optional[Callable] = None,
        project: Optional[str] = None,
        transform: Optional[Callable] = None,
        local_cache_dir: Optional[str] = None,
    ):
        """
        Initialize GCS dataset.
        
        Args:
            manifest_path: Path to manifest (local or gs://bucket/path)
            gcs_bucket: GCS bucket name
            gcs_prefix: Prefix for audio files in bucket
            sample_rate: Target sample rate
            max_length_sec: Maximum audio length
            feature_extractor: Optional feature extraction function
            project: GCP project ID
            transform: Optional transform
            local_cache_dir: Local directory to cache downloaded files
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage is required for GCS access. Install with: pip install google-cloud-storage")
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio loading")
        
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix.strip("/")
        self.sample_rate = sample_rate
        self.max_length = int(max_length_sec * sample_rate)
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
        
        # Initialize GCS client
        self.gcs_client = gcs_storage.Client(project=project)
        self.bucket = self.gcs_client.bucket(gcs_bucket)
        
        # Load manifest
        self.manifest = self._load_manifest(manifest_path)
        
        # Setup local cache
        if self.local_cache_dir:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_manifest(self, manifest_path: str) -> list[dict]:
        """Load manifest from local file or GCS."""
        if manifest_path.startswith("gs://"):
            # Parse GCS URI
            parts = manifest_path.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1] if len(parts) > 1 else ""
            
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            manifest = json.loads(blob.download_as_string().decode("utf-8"))
        else:
            with open(manifest_path) as f:
                manifest = json.load(f)
        
        # Normalize manifest format (handle both 'path' and 'audio_path' keys)
        for item in manifest:
            # Handle path key
            if "path" in item and "audio_path" not in item:
                item["audio_path"] = item["path"]
            # Handle emotion -> emotion_idx conversion
            if "emotion" in item and "emotion_idx" not in item:
                emotion = item["emotion"].lower()
                item["emotion_idx"] = EMOTION_LABELS.index(emotion) if emotion in EMOTION_LABELS else 0
        
        return manifest
    
    def _download_from_gcs(self, gcs_path: str) -> bytes:
        """Download file from GCS."""
        blob = self.bucket.blob(gcs_path)
        return blob.download_as_bytes()
    
    def _get_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """Get audio from GCS or local cache."""
        # Handle full gs:// URIs
        if audio_path.startswith("gs://"):
            parts = audio_path.replace("gs://", "").split("/", 1)
            gcs_path = parts[1] if len(parts) > 1 else ""
        else:
            gcs_path = f"{self.gcs_prefix}/{audio_path}" if self.gcs_prefix else audio_path
        
        # Check local cache
        if self.local_cache_dir:
            local_path = self.local_cache_dir / audio_path.replace("/", "_")
            if local_path.exists():
                return librosa.load(local_path, sr=self.sample_rate)
        
        # Download from GCS
        audio_bytes = self._download_from_gcs(gcs_path)
        
        # Save to local cache
        if self.local_cache_dir:
            local_path = self.local_cache_dir / audio_path.replace("/", "_")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(audio_bytes)
        
        # Load audio from bytes
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate)
        return audio, sr
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.manifest[idx]
        
        # Load audio from GCS
        audio, sr = self._get_audio(item["audio_path"])
        
        # Pad or truncate
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        
        # Extract features
        if self.feature_extractor:
            features = self.feature_extractor(audio, sr)
        else:
            features = {"audio": torch.from_numpy(audio).float()}
        
        # Add label
        features["emotion_idx"] = torch.tensor(item["emotion_idx"], dtype=torch.long)
        
        # Apply transform
        if self.transform:
            features = self.transform(features)
        
        return features
    
    @property
    def num_classes(self) -> int:
        return len(EMOTION_LABELS)


class ProsodyFeatureExtractorWrapper:
    """
    Wrapper around ProsodyFeatureExtractor for dataset use.
    
    Extracts prosody and phonetic features and returns as tensors.
    Phonetic features are estimated from audio spectral properties when
    no transcript is available.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        include_phonetic: bool = True,
    ):
        from prosody_ssm.features import ProsodyFeatureExtractor
        
        self.prosody_extractor = ProsodyFeatureExtractor(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
        )
        self.sample_rate = sample_rate
        self.include_phonetic = include_phonetic
    
    def _estimate_phonetic_from_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Estimate phonetic features directly from audio when no transcript is available.
        Uses spectral properties to approximate vowel/consonant ratios and syllable count.
        
        Returns: np.ndarray of shape (4,) [vowel_ratio, consonant_ratio, stressed_syllables, phoneme_count]
        """
        import librosa
        
        # Estimate syllable count from energy peaks (proxy for phoneme count)
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        if len(rms) < 3:
            return np.array([0.5, 0.5, 0.0, 0.0])
        
        # Smooth and find peaks for syllable estimation
        from scipy.signal import find_peaks
        rms_smooth = np.convolve(rms, np.ones(3)/3, mode='same')
        peaks, properties = find_peaks(rms_smooth, height=np.mean(rms_smooth) * 0.5, distance=4)
        syllable_count = max(len(peaks), 1)
        
        # Estimate vowel/consonant ratio from spectral flatness
        # Vowels have more harmonic (less flat) spectra; consonants are noisier (more flat)
        flatness = librosa.feature.spectral_flatness(y=audio, hop_length=512)[0]
        mean_flatness = float(np.mean(flatness))
        
        # Higher flatness = more consonant-like frames
        vowel_ratio = max(0.0, min(1.0, 1.0 - mean_flatness * 3))
        consonant_ratio = 1.0 - vowel_ratio
        
        # Estimate stressed syllables from energy variance in peaks
        if len(peaks) > 1:
            peak_energies = rms[peaks]
            energy_std = float(np.std(peak_energies))
            energy_mean = float(np.mean(peak_energies)) + 1e-8
            # High energy variance suggests more stressed/unstressed contrast
            stressed_ratio = min(1.0, energy_std / energy_mean)
            stressed_count = max(1, int(syllable_count * stressed_ratio))
        else:
            stressed_count = 1
        
        return np.array([vowel_ratio, consonant_ratio, float(stressed_count), float(syllable_count)])
    
    def __call__(self, audio: np.ndarray, sr: int) -> dict:
        """Extract features from audio."""
        prosody_features = self.prosody_extractor.extract(audio, sr)
        prosody_vec = prosody_features.to_vector()
        
        result = {
            "prosody_features": torch.from_numpy(prosody_vec).float().unsqueeze(0),  # (1, prosody_dim)
        }
        
        if self.include_phonetic:
            phonetic_vec = self._estimate_phonetic_from_audio(audio, sr)
            result["phonetic_features"] = torch.from_numpy(phonetic_vec).float().unsqueeze(0)
        else:
            result["phonetic_features"] = torch.zeros(1, 4)
        
        return result


def create_dataloader(
    manifest_path: Union[str, Path],
    data_root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    sample_rate: int = 16000,
    max_length_sec: float = 10.0,
    use_feature_extractor: bool = True,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        manifest_path: Path to manifest JSON
        data_root: Root directory for audio files
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        sample_rate: Audio sample rate
        max_length_sec: Max audio length in seconds
        use_feature_extractor: Whether to extract features
        pin_memory: Pin memory for faster GPU transfer
        **kwargs: Additional args for DataLoader
    
    Returns:
        DataLoader instance
    """
    feature_extractor = None
    if use_feature_extractor:
        feature_extractor = ProsodyFeatureExtractorWrapper(sample_rate=sample_rate)
    
    dataset = EmotionSpeechDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        sample_rate=sample_rate,
        max_length_sec=max_length_sec,
        feature_extractor=feature_extractor,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_s3_dataloader(
    manifest_path: str,
    s3_bucket: str,
    s3_prefix: str = "",
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    sample_rate: int = 16000,
    max_length_sec: float = 10.0,
    use_feature_extractor: bool = True,
    local_cache_dir: Optional[str] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader that streams from S3.
    
    Args:
        manifest_path: Path to manifest (local or s3://)
        s3_bucket: S3 bucket name
        s3_prefix: Prefix for audio files
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        sample_rate: Audio sample rate
        max_length_sec: Max audio length
        use_feature_extractor: Whether to extract features
        local_cache_dir: Local cache directory
        **kwargs: Additional DataLoader args
    
    Returns:
        DataLoader instance
    """
    feature_extractor = None
    if use_feature_extractor:
        feature_extractor = ProsodyFeatureExtractorWrapper(sample_rate=sample_rate)
    
    dataset = S3EmotionSpeechDataset(
        manifest_path=manifest_path,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        sample_rate=sample_rate,
        max_length_sec=max_length_sec,
        feature_extractor=feature_extractor,
        local_cache_dir=local_cache_dir,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )


def create_gcs_dataloader(
    manifest_path: str,
    gcs_bucket: str,
    gcs_prefix: str = "",
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    sample_rate: int = 16000,
    max_length_sec: float = 10.0,
    use_feature_extractor: bool = True,
    local_cache_dir: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader that streams from Google Cloud Storage.
    
    Args:
        manifest_path: Path to manifest (local or gs://)
        gcs_bucket: GCS bucket name
        gcs_prefix: Prefix for audio files
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        sample_rate: Audio sample rate
        max_length_sec: Max audio length
        use_feature_extractor: Whether to extract features
        local_cache_dir: Local cache directory
        project: GCP project ID
        **kwargs: Additional DataLoader args
    
    Returns:
        DataLoader instance
    """
    feature_extractor = None
    if use_feature_extractor:
        feature_extractor = ProsodyFeatureExtractorWrapper(sample_rate=sample_rate)
    
    dataset = GCSEmotionSpeechDataset(
        manifest_path=manifest_path,
        gcs_bucket=gcs_bucket,
        gcs_prefix=gcs_prefix,
        sample_rate=sample_rate,
        max_length_sec=max_length_sec,
        feature_extractor=feature_extractor,
        local_cache_dir=local_cache_dir,
        project=project,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )


class FeedbackDataset(Dataset):
    """
    Per-utterance feedback dataset for fine-tuning emotion models.

    Reads a JSON manifest where each sample contains pre-extracted prosody
    and phonetic features, an emotion label, an optional sample weight, and
    optional VAD (valence-arousal-dominance) regression targets.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
    ):
        """
        Initialize feedback dataset.

        Args:
            manifest_path: Path to JSON manifest file. Each entry should have:
                - prosody_features (list[float])
                - phonetic_features (list[float])
                - emotion_idx (int)
                - sample_weight (float, optional, default 1.0)
                - vad_targets (list[float] of length 3, optional)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for dataset classes")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        item = self.manifest[idx]

        prosody = torch.tensor(item["prosody_features"], dtype=torch.float32).unsqueeze(0)
        phonetic = torch.tensor(item["phonetic_features"], dtype=torch.float32).unsqueeze(0)
        emotion_idx = torch.tensor(item["emotion_idx"], dtype=torch.long)
        sample_weight = torch.tensor(item.get("sample_weight", 1.0), dtype=torch.float32)

        if "vad_targets" in item and item["vad_targets"] is not None:
            vad_targets = torch.tensor(item["vad_targets"], dtype=torch.float32)
        else:
            vad_targets = torch.zeros(3, dtype=torch.float32)

        return {
            "prosody_features": prosody,
            "phonetic_features": phonetic,
            "emotion_idx": emotion_idx,
            "sample_weight": sample_weight,
            "vad_targets": vad_targets,
        }


class ConversationFeedbackDataset(Dataset):
    """
    Session-level conversation dataset for ConversationPredictor training.

    Reads a JSON manifest where each sample represents a conversation session
    with a sequence of per-utterance emotion/VAD/confidence features and
    associated prediction targets (escalation, churn, CSAT, etc.).
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        max_window: int = 20,
    ):
        """
        Initialize conversation feedback dataset.

        Args:
            manifest_path: Path to JSON manifest file. Each entry should have:
                - session_id (str)
                - utterances (list[dict]) each with:
                    - emotion_probs (list[float] of length 8)
                    - vad (list[float] of length 3)
                    - confidence (float)
                - targets (dict) with optional keys such as:
                    will_escalate, churn_risk, resolution_prob,
                    deal_close_prob, intervention_needed, final_csat,
                    sentiment_forecast, recommended_tone
            max_window: Maximum sequence length. Shorter sequences are
                zero-padded; longer ones are truncated (last *max_window*
                utterances are kept).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for dataset classes")

        self.max_window = max_window

        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        item = self.manifest[idx]
        utterances = item["utterances"]

        # Build per-utterance feature vectors: emotion_probs(8) + vad(3) + confidence(1) = 12
        seq: list[list[float]] = []
        for utt in utterances:
            vec = list(utt["emotion_probs"]) + list(utt["vad"]) + [utt["confidence"]]
            seq.append(vec)

        actual_length = len(seq)

        # Truncate (keep last max_window utterances) or pad with zeros
        if actual_length > self.max_window:
            seq = seq[-self.max_window:]
            actual_length = self.max_window

        features = torch.zeros(self.max_window, 12, dtype=torch.float32)
        if actual_length > 0:
            features[:actual_length] = torch.tensor(seq, dtype=torch.float32)

        result: dict = {
            "features": features,
            "length": torch.tensor(actual_length, dtype=torch.long),
        }

        # Map each target key to a tensor
        targets = item.get("targets", {})
        for key, value in targets.items():
            if isinstance(value, list):
                result[key] = torch.tensor(value, dtype=torch.float32)
            else:
                result[key] = torch.tensor(value, dtype=torch.float32)

        return result


def create_feedback_dataloader(
    manifest_path: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for per-utterance feedback training.

    Args:
        manifest_path: Path to feedback manifest JSON
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        **kwargs: Additional args for DataLoader

    Returns:
        DataLoader instance
    """
    dataset = FeedbackDataset(manifest_path=manifest_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_conversation_feedback_dataloader(
    manifest_path: Union[str, Path],
    max_window: int = 20,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for session-level conversation feedback training.

    Args:
        manifest_path: Path to conversation feedback manifest JSON
        max_window: Maximum utterance sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        **kwargs: Additional args for DataLoader

    Returns:
        DataLoader instance
    """
    dataset = ConversationFeedbackDataset(
        manifest_path=manifest_path,
        max_window=max_window,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )
