from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--clip-start", type=float, default=0.0)
    parser.add_argument("--clip-end", type=float, default=0.0)
    args = parser.parse_args()

    import stable_whisper

    transcript = Path(args.transcript).read_text(encoding="utf-8")
    model = stable_whisper.load_model(args.model)
    result = model.align(args.media, transcript, language="en", original_split=True)
    payload = result.to_dict() if hasattr(result, "to_dict") else result
    transcribe_kwargs = {"language": "en", "word_timestamps": True}
    if args.clip_end > args.clip_start:
        transcribe_kwargs["clip_timestamps"] = [args.clip_start, args.clip_end]
    hypothesis = model.transcribe(args.media, **transcribe_kwargs)
    payload["audio_hypothesis"] = str(getattr(hypothesis, "text", "") or "").strip()
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
