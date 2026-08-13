from __future__ import annotations

import argparse
from pathlib import Path

import torch
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from .audio import load_audio, save_audio, speaker_embedding


def synthesize(
    model_path: str,
    text: str,
    reference_audio: str,
    output_path: str,
    vocoder_name: str = "microsoft/speecht5_hifigan",
    speaker_encoder_name: str = "speechbrain/spkrec-xvect-voxceleb",
) -> Path:
    if not text.strip():
        raise ValueError("Text cannot be blank")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SpeechT5Processor.from_pretrained(model_path)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_path).to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name).to(device).eval()
    speaker_model = EncoderClassifier.from_hparams(
        source=speaker_encoder_name,
        run_opts={"device": str(device)},
        savedir=str(Path(".cache/speechbrain") / speaker_encoder_name.replace("/", "--")),
    )

    waveform = load_audio(reference_audio, 16000)
    embedding = torch.tensor(
        speaker_embedding(speaker_model, waveform), dtype=torch.float32, device=device
    ).unsqueeze(0)
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.inference_mode():
        speech = model.generate_speech(input_ids, embedding, vocoder=vocoder)

    output = Path(output_path).resolve()
    save_audio(output, speech.cpu().numpy(), 16000)
    print(f"Wrote {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize speech with a fine-tuned SpeechT5 model"
    )
    parser.add_argument("--model", required=True, help="Local checkpoint or Hugging Face model id")
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--output", default="outputs/generated.wav")
    parser.add_argument("--vocoder", default="microsoft/speecht5_hifigan")
    parser.add_argument("--speaker-encoder", default="speechbrain/spkrec-xvect-voxceleb")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    synthesize(
        model_path=args.model,
        text=args.text,
        reference_audio=args.reference_audio,
        output_path=args.output,
        vocoder_name=args.vocoder,
        speaker_encoder_name=args.speaker_encoder,
    )


if __name__ == "__main__":
    main()
