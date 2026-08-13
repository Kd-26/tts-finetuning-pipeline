# Indian Multilingual Speech-to-Text Options

Last reviewed: 13 August 2026.

This is a practical shortlist of strong speech-to-text (STT/ASR) options for Indian
languages. It is not a universal accuracy ranking: results vary substantially by language,
dialect, microphone, background noise, code-mixing, and domain vocabulary. Benchmark the
final candidates on representative recordings before choosing one.

## Quick recommendations

| Requirement | Recommended starting point |
|---|---|
| Managed API focused on Indian languages and code-mixing | Sarvam AI Saaras v3 |
| Open model covering all 22 scheduled Indian languages | AI4Bharat IndicConformer 600M Multilingual |
| Managed cloud STT with broader platform integration | Google Cloud Speech-to-Text or Azure AI Speech |
| General-purpose, offline multilingual baseline | OpenAI Whisper large-v3 or large-v3-turbo |

## Leading options

### 1. Sarvam AI Saaras v3

Saaras v3 is the strongest first candidate when the application is India-focused and a
managed API is acceptable. It supports all 22 scheduled Indian languages plus Indian
English, automatic language detection, dialects, and code-mixed speech. Its output modes
include native transcription, English translation, verbatim transcription, transliteration,
and code-mixed output.

- Best fit: multilingual voice agents, call-centre audio, captions, and mixed Indian
  language/English conversations.
- Interfaces: synchronous REST, batch processing, and WebSocket streaming.
- Important limit: synchronous REST accepts up to 30 seconds; batch processing supports
  files up to two hours.
- Trade-off: it is a hosted service, so evaluate pricing, latency, regional availability,
  retention, and data-processing terms for the deployment.
- Official documentation: [Saaras v3 model](https://docs.sarvam.ai/api/getting-started/models/saaras)
  and [STT API overview](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/overview).

### 2. AI4Bharat IndicConformer 600M Multilingual

IndicConformer is the strongest open, India-specific starting point. The 600M multilingual
model covers Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani,
Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi,
Tamil, Telugu, and Urdu. It uses a hybrid CTC/RNNT architecture and expects 16 kHz audio.

- Best fit: self-hosted or offline transcription, research, private data, and systems that
  need all 22 scheduled Indian languages without a per-request API dependency.
- License: MIT.
- Trade-off: a 600M-parameter model needs more compute and operational work than an API.
  Access to the model files currently requires accepting the repository conditions, and
  production streaming/serving must be built around the model.
- Official model card: [AI4Bharat IndicConformer 600M Multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual).

### 3. Google Cloud Speech-to-Text V2 (Chirp)

Google Cloud STT is a good managed option when the application already uses Google Cloud or
needs features such as punctuation, diarization, model adaptation, and batch/streaming
infrastructure. Chirp models support many major Indian locales, including Indian English,
Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sindhi, Tamil,
and Telugu, but model features and availability vary by locale and region.

- Best fit: managed enterprise workloads and teams already operating on Google Cloud.
- Trade-off: do not assume that one Chirp version or region supports every target language
  and feature. Verify each locale/model/region combination before implementation.
- Official documentation: [Speech-to-Text V2 supported languages](https://cloud.google.com/speech-to-text/v2/docs/speech-to-text-supported-languages).

### 4. Azure AI Speech

Azure AI Speech is another strong managed candidate for applications already built on Azure.
Its STT language table includes major Indian locales and the service provides SDK, REST,
batch, and real-time integration paths. Feature support can differ by locale and region.

- Best fit: Microsoft-centric enterprise environments, compliance-controlled deployments,
  and applications that also use Azure language or contact-centre services.
- Trade-off: validate the exact locale, region, diarization, customization, and container
  requirements instead of treating general language support as feature parity.
- Official documentation: [Azure Speech language support](https://learn.microsoft.com/azure/ai-services/speech-service/language-support?tabs=stt-tts).

### 5. OpenAI Whisper large-v3 / large-v3-turbo

Whisper remains a useful general-purpose open multilingual baseline. Its language set includes
many Indian languages such as Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam,
Marathi, Nepali, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, and Urdu. The turbo checkpoint is
faster, but it is intended for transcription rather than speech translation.

- Best fit: offline prototypes, a common cross-language baseline, and deployments that also
  need many non-Indian languages.
- Trade-off: it is not India-specific, accuracy is uneven across languages and accents, and
  weak or silent audio can cause hallucinated text. For translation to English, use a
  multilingual non-turbo Whisper model.
- Official sources: [OpenAI Whisper repository](https://github.com/openai/whisper) and
  [Whisper large-v3-turbo model card](https://huggingface.co/openai/whisper-large-v3-turbo).

## How to choose

Test at least 30–100 manually transcribed clips per target language, sampled from the actual
deployment conditions. Include code-mixed speech, names, numbers, domain terms, different
accents, noisy audio, and telephony audio where relevant. Compare:

- word error rate (WER) and character error rate (CER) per language;
- proper-name, number, and domain-term accuracy;
- language-detection and code-switch accuracy;
- streaming partial-result stability and finalization latency;
- real-time factor, throughput, availability, and total cost;
- data retention, residency, consent, and deletion controls.

Do not select a system from a single vendor-reported aggregate score. A model can lead on
Hindi studio speech and still perform poorly on Tamil telephony audio or Hindi-English
code-mixing.

## Using STT with this TTS pipeline

STT can bootstrap transcripts for a consented TTS dataset or check synthesized outputs, but
its output should not be used as training truth without human review. Recognition mistakes
become label noise and directly reduce TTS pronunciation and intelligibility.

Before adding an STT-generated transcript to the manifest:

1. Have a fluent reviewer compare it with the recording.
2. Correct names, punctuation, code-mixed words, and script choice consistently.
3. Expand digits into words because this pipeline's validator rejects digits.
4. Keep the transcript in the `text` column and preserve the required manifest schema
   described in [DATA.md](DATA.md).

