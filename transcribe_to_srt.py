#!/usr/bin/env python3
"""Transcribe an audio/video file with faster-whisper and write an SRT."""

# This script uses the faster-whisper library to transcribe an audio or video file and outputs the transcription in SRT subtitle format. It handles word-level timestamps, groups words into readable cues, and formats them according to SRT specifications.
pass
import argparse
import textwrap
from pathlib import Path

import av
from faster_whisper import WhisperModel


# PyAV 19 removed an argument still passed by some faster-whisper releases.
_av_open = av.open


# Retry without that obsolete argument only for this compatibility mismatch.
def _open_compatible(*args, **kwargs):
    try:
        return _av_open(*args, **kwargs)
    except TypeError as error:
        if "metadata_errors" not in kwargs or "metadata_errors" not in str(error):
            raise
        kwargs.pop("metadata_errors")
        return _av_open(*args, **kwargs)


av.open = _open_compatible


# Convert seconds to the comma-millisecond timestamp format required by SRT.
def format_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


# Normalize whitespace, then place longer captions on a balanced line break.
def wrap_caption(text: str, width: int = 42) -> str:
    """Wrap onto at most two balanced lines."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text

    breakpoints = [
        i for i, char in enumerate(text)
        if char == " " and i <= width and len(text) - i - 1 <= width
    ]
    if not breakpoints:
        return textwrap.fill(text, width=width, break_long_words=False)

    midpoint = len(text) / 2
    split_at = min(breakpoints, key=lambda i: abs(i - midpoint))
    return f"{text[:split_at]}\n{text[split_at + 1:]}"


# Group timestamped words into readable cues and write numbered SRT blocks.
def write_srt(segments, output_path: Path) -> int:
    # Keep only words with usable text and both timing boundaries.
    cues = []
    for segment in segments:
        words = [
            word for word in (segment.words or [])
            if word.word.strip() and word.start is not None and word.end is not None
        ]
        # Split each segment into cues capped by word count and character count.
        group = []
        for word in words:
            candidate = group + [word]
            candidate_text = "".join(item.word for item in candidate).strip()
            if group and (len(candidate) > 8 or len(candidate_text) > 70):
                cues.append(group)
                group = [word]
            else:
                group = candidate
        if group:
            cues.append(group)

    # Use the earliest start and latest end to cover all words in each cue.
    blocks = []
    for number, group in enumerate(cues, start=1):
        caption = "".join(word.word for word in group).strip()
        start = min(word.start for word in group)
        end = max(word.end for word in group)
        if end <= start:
            end = start + 0.25
        blocks.append(
            f"{number}\n{format_timestamp(start)} --> {format_timestamp(end)}\n"
            f"{wrap_caption(caption)}"
        )

    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return len(blocks)


# Parse options, run transcription, and save the resulting subtitles.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input audio or video file")
    parser.add_argument("output", type=Path, nargs="?", help="Output SRT path")
    parser.add_argument("--model", default="small.en", help="Whisper model (default: small.en)")
    parser.add_argument("--language", default="en", help="Audio language code (default: en)")
    args = parser.parse_args()

    # Default to the input filename with an .srt extension.
    output_path = args.output or args.input.with_suffix(".srt")
    # Load the selected model and request word-level timings for cue creation.
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(args.input),
        language=args.language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )
    cue_count = write_srt(segments, output_path)
    print(f"Wrote {cue_count} subtitle cues to {output_path}")


# Run the CLI entry point only when this file is executed directly.
if __name__ == "__main__":
    main()
