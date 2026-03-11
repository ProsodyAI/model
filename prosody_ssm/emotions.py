"""
Emotion annotation system for transcribed text.

Provides utilities to annotate text with emotion labels,
prosodic markers, and confidence scores for LLM consumption.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Emotion(str, Enum):
    """Emotion categories with semantic groupings."""

    # Positive emotions
    HAPPY = "happy"
    EXCITED = "excited"
    AMUSED = "amused"
    CONTENT = "content"

    # Negative emotions
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    CONTEMPT = "contempt"
    ANXIOUS = "anxious"

    # Neutral/Other
    NEUTRAL = "neutral"
    SURPRISED = "surprised"
    CONFUSED = "confused"

    @property
    def valence(self) -> float:
        """Get typical valence for this emotion (-1 to +1)."""
        valence_map = {
            self.HAPPY: 0.8, self.EXCITED: 0.7, self.AMUSED: 0.6, self.CONTENT: 0.5,
            self.SAD: -0.7, self.ANGRY: -0.6, self.FEARFUL: -0.7,
            self.DISGUSTED: -0.5, self.CONTEMPT: -0.4, self.ANXIOUS: -0.5,
            self.NEUTRAL: 0.0, self.SURPRISED: 0.2, self.CONFUSED: -0.1,
        }
        return valence_map.get(self, 0.0)

    @property
    def arousal(self) -> float:
        """Get typical arousal for this emotion (0 to 1)."""
        arousal_map = {
            self.HAPPY: 0.7, self.EXCITED: 0.9, self.AMUSED: 0.5, self.CONTENT: 0.3,
            self.SAD: 0.3, self.ANGRY: 0.9, self.FEARFUL: 0.8,
            self.DISGUSTED: 0.6, self.CONTEMPT: 0.4, self.ANXIOUS: 0.7,
            self.NEUTRAL: 0.3, self.SURPRISED: 0.8, self.CONFUSED: 0.5,
        }
        return arousal_map.get(self, 0.5)


@dataclass
class EmotionSpan:
    """An emotion annotation for a text span."""

    start: int  # Character start index
    end: int    # Character end index
    text: str   # The text content
    emotion: Emotion
    confidence: float

    # Optional prosodic markers
    pitch_trend: Optional[str] = None  # "rising", "falling", "flat", "varied"
    intensity: Optional[str] = None    # "soft", "normal", "loud", "emphasized"
    tempo: Optional[str] = None        # "slow", "normal", "fast"

    def __str__(self) -> str:
        """String representation with emotion tag."""
        return f"[{self.emotion.value}:{self.confidence:.2f}]{self.text}[/{self.emotion.value}]"


@dataclass
class AnnotatedTranscript:
    """A fully annotated transcript with emotion and prosody information."""

    raw_text: str
    emotion_spans: list[EmotionSpan] = field(default_factory=list)
    overall_emotion: Optional[Emotion] = None
    overall_confidence: float = 0.0

    # Aggregate prosodic information
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    # Metadata
    duration_seconds: float = 0.0
    word_count: int = 0

    def to_annotated_text(self, format: str = "xml") -> str:
        """
        Convert to annotated text format.

        Args:
            format: Output format - "xml", "markdown", or "inline"

        Returns:
            Annotated text string
        """
        if format == "xml":
            return self._to_xml()
        elif format == "markdown":
            return self._to_markdown()
        elif format == "inline":
            return self._to_inline()
        else:
            raise ValueError(f"Unknown format: {format}")

    def _to_xml(self) -> str:
        """Convert to XML-style annotation format."""
        if not self.emotion_spans:
            return f'<speech emotion="{self.overall_emotion.value if self.overall_emotion else "neutral"}">{self.raw_text}</speech>'

        # Build annotated text
        result = []
        last_end = 0

        for span in sorted(self.emotion_spans, key=lambda s: s.start):
            # Add unannotated text before this span
            if span.start > last_end:
                result.append(self.raw_text[last_end:span.start])

            # Add annotated span
            attrs = [f'emotion="{span.emotion.value}"', f'confidence="{span.confidence:.2f}"']
            if span.pitch_trend:
                attrs.append(f'pitch="{span.pitch_trend}"')
            if span.intensity:
                attrs.append(f'intensity="{span.intensity}"')
            if span.tempo:
                attrs.append(f'tempo="{span.tempo}"')

            attr_str = " ".join(attrs)
            result.append(f"<segment {attr_str}>{span.text}</segment>")
            last_end = span.end

        # Add remaining text
        if last_end < len(self.raw_text):
            result.append(self.raw_text[last_end:])

        # Wrap in speech element
        speech_attrs = [
            f'overall_emotion="{self.overall_emotion.value if self.overall_emotion else "neutral"}"',
            f'valence="{self.valence:.2f}"',
            f'arousal="{self.arousal:.2f}"',
            f'dominance="{self.dominance:.2f}"',
        ]

        return f'<speech {" ".join(speech_attrs)}>{"".join(result)}</speech>'

    def _to_markdown(self) -> str:
        """Convert to markdown format with emotion markers."""
        lines = [
            f"**Overall Emotion:** {self.overall_emotion.value if self.overall_emotion else 'neutral'} "
            f"(confidence: {self.overall_confidence:.0%})",
            f"**Emotional Tone:** valence={self.valence:+.2f}, arousal={self.arousal:.2f}, dominance={self.dominance:.2f}",
            "",
            "**Transcription:**",
        ]

        if not self.emotion_spans:
            lines.append(f"> {self.raw_text}")
        else:
            # Build text with inline annotations
            annotated = self.raw_text
            offset = 0

            for span in sorted(self.emotion_spans, key=lambda s: s.start):
                marker = f" *[{span.emotion.value}]*"
                insert_pos = span.end + offset
                annotated = annotated[:insert_pos] + marker + annotated[insert_pos:]
                offset += len(marker)

            lines.append(f"> {annotated}")

        return "\n".join(lines)

    def _to_inline(self) -> str:
        """Convert to simple inline format for LLM prompts."""
        if not self.emotion_spans:
            emotion = self.overall_emotion.value if self.overall_emotion else "neutral"
            return f"[EMOTION: {emotion}] {self.raw_text}"

        # Group consecutive spans with same emotion
        result = []
        current_text = []
        current_emotion = None

        last_end = 0
        for span in sorted(self.emotion_spans, key=lambda s: s.start):
            # Add gap text to current segment
            if span.start > last_end:
                gap_text = self.raw_text[last_end:span.start]
                if current_text:
                    current_text.append(gap_text)
                else:
                    result.append(gap_text)

            # Check if emotion changed
            if current_emotion != span.emotion:
                # Flush current segment
                if current_text and current_emotion:
                    result.append(f"[{current_emotion.value.upper()}] {''.join(current_text)} [/{current_emotion.value.upper()}]")
                elif current_text:
                    result.append(''.join(current_text))

                current_text = [span.text]
                current_emotion = span.emotion
            else:
                current_text.append(span.text)

            last_end = span.end

        # Flush remaining
        if current_text and current_emotion:
            result.append(f"[{current_emotion.value.upper()}] {''.join(current_text)} [/{current_emotion.value.upper()}]")
        elif current_text:
            result.append(''.join(current_text))

        # Add any trailing text
        if last_end < len(self.raw_text):
            result.append(self.raw_text[last_end:])

        return ''.join(result)

    def to_llm_context(self) -> str:
        """
        Generate context string optimized for LLM consumption.

        Provides the LLM with emotional context about the speaker
        to inform response generation.
        """
        emotion = self.overall_emotion.value if self.overall_emotion else "neutral"

        # Determine emotional descriptors
        mood_descriptors = []

        if self.valence > 0.3:
            mood_descriptors.append("positive")
        elif self.valence < -0.3:
            mood_descriptors.append("negative")

        if self.arousal > 0.7:
            mood_descriptors.append("highly energetic")
        elif self.arousal < 0.3:
            mood_descriptors.append("calm")

        if self.dominance > 0.7:
            mood_descriptors.append("assertive")
        elif self.dominance < 0.3:
            mood_descriptors.append("uncertain")

        mood_str = ", ".join(mood_descriptors) if mood_descriptors else "neutral"

        # Build context
        context = f"""[SPEAKER EMOTION ANALYSIS]
Primary Emotion: {emotion} (confidence: {self.overall_confidence:.0%})
Emotional State: {mood_str}
Valence: {self.valence:+.2f} (negative ← → positive)
Arousal: {self.arousal:.2f} (calm ← → excited)
Dominance: {self.dominance:.2f} (submissive ← → dominant)

[TRANSCRIPTION WITH PROSODY]
{self.to_annotated_text('inline')}

[END EMOTION ANALYSIS]"""

        return context


class EmotionAnnotator:
    """
    Annotates transcribed text with emotion labels based on prosody analysis.

    Combines utterance-level and word/phrase-level emotion detection
    to produce rich annotations for LLM consumption.
    """

    def __init__(
        self,
        segment_by_punctuation: bool = True,
        min_segment_words: int = 3,
    ):
        """
        Initialize the annotator.

        Args:
            segment_by_punctuation: Whether to segment text by punctuation
            min_segment_words: Minimum words per segment for annotation
        """
        self.segment_by_punctuation = segment_by_punctuation
        self.min_segment_words = min_segment_words

        # Punctuation patterns for segmentation
        self._sentence_pattern = re.compile(r'([.!?]+)\s*')
        self._clause_pattern = re.compile(r'([,;:—–-])\s*')

    def annotate(
        self,
        text: str,
        emotion: Emotion,
        confidence: float,
        valence: float = 0.0,
        arousal: float = 0.5,
        dominance: float = 0.5,
        word_emotions: Optional[list[tuple[str, Emotion, float]]] = None,
        prosody_markers: Optional[dict] = None,
    ) -> AnnotatedTranscript:
        """
        Create an annotated transcript.

        Args:
            text: The transcribed text
            emotion: Primary detected emotion
            confidence: Confidence in the primary emotion
            valence: Overall valence (-1 to +1)
            arousal: Overall arousal (0 to 1)
            dominance: Overall dominance (0 to 1)
            word_emotions: Optional per-word emotion labels
            prosody_markers: Optional prosody information (pitch, intensity, tempo)

        Returns:
            AnnotatedTranscript with emotion spans
        """
        transcript = AnnotatedTranscript(
            raw_text=text,
            overall_emotion=emotion,
            overall_confidence=confidence,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            word_count=len(text.split()),
        )

        # If we have word-level emotions, create spans
        if word_emotions:
            transcript.emotion_spans = self._create_word_spans(text, word_emotions)
        elif self.segment_by_punctuation:
            # Create sentence-level spans
            transcript.emotion_spans = self._create_sentence_spans(
                text, emotion, confidence, prosody_markers
            )
        else:
            # Single span for entire text
            transcript.emotion_spans = [
                EmotionSpan(
                    start=0,
                    end=len(text),
                    text=text,
                    emotion=emotion,
                    confidence=confidence,
                    **(prosody_markers or {}),
                )
            ]

        return transcript

    def _create_word_spans(
        self,
        text: str,
        word_emotions: list[tuple[str, Emotion, float]],
    ) -> list[EmotionSpan]:
        """Create spans from word-level emotion labels."""
        spans = []
        current_pos = 0

        for word, emotion, confidence in word_emotions:
            # Find word in text
            word_start = text.find(word, current_pos)
            if word_start == -1:
                continue

            word_end = word_start + len(word)

            spans.append(EmotionSpan(
                start=word_start,
                end=word_end,
                text=word,
                emotion=emotion,
                confidence=confidence,
            ))

            current_pos = word_end

        return spans

    def _create_sentence_spans(
        self,
        text: str,
        default_emotion: Emotion,
        default_confidence: float,
        prosody_markers: Optional[dict] = None,
    ) -> list[EmotionSpan]:
        """Create spans from sentence segmentation."""
        spans = []

        # Split by sentence-ending punctuation
        sentences = self._sentence_pattern.split(text)

        current_pos = 0
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if not sentence.strip():
                current_pos += len(sentence)
                continue

            # Add punctuation if present
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            # Skip if too short
            if len(sentence.split()) < self.min_segment_words:
                current_pos += len(sentence)
                continue

            start = text.find(sentence.strip(), current_pos)
            if start == -1:
                start = current_pos

            end = start + len(sentence.strip())

            span = EmotionSpan(
                start=start,
                end=end,
                text=sentence.strip(),
                emotion=default_emotion,
                confidence=default_confidence,
            )

            if prosody_markers:
                span.pitch_trend = prosody_markers.get('pitch_trend')
                span.intensity = prosody_markers.get('intensity')
                span.tempo = prosody_markers.get('tempo')

            spans.append(span)
            current_pos = end

        return spans

    def merge_spans(
        self,
        spans: list[EmotionSpan],
        merge_threshold: float = 0.8,
    ) -> list[EmotionSpan]:
        """
        Merge adjacent spans with similar emotions.

        Args:
            spans: List of emotion spans
            merge_threshold: Confidence threshold for merging

        Returns:
            Merged list of spans
        """
        if not spans:
            return []

        merged = [spans[0]]

        for span in spans[1:]:
            last = merged[-1]

            # Merge if same emotion and high confidence
            if (last.emotion == span.emotion and
                last.confidence >= merge_threshold and
                span.confidence >= merge_threshold and
                span.start <= last.end + 5):  # Allow small gaps

                # Extend the last span
                gap_text = "" if span.start <= last.end else " "
                merged[-1] = EmotionSpan(
                    start=last.start,
                    end=span.end,
                    text=last.text + gap_text + span.text,
                    emotion=last.emotion,
                    confidence=(last.confidence + span.confidence) / 2,
                    pitch_trend=last.pitch_trend or span.pitch_trend,
                    intensity=last.intensity or span.intensity,
                    tempo=last.tempo or span.tempo,
                )
            else:
                merged.append(span)

        return merged
