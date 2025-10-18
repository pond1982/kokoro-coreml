# Kokoro CoreML TTS – iOS Sample App

This directory contains a minimal SwiftUI app that demonstrates end‑to‑end text‑to‑speech using Kokoro's Core ML exports. The goal is to provide a self‑contained starting point for measuring latency, memory consumption, and user experience of on‑device synthesis.

## Requirements

- Xcode 15.4 or later
- iOS 17 simulator or device (Apple Silicon strongly recommended)
- Swift 5.9 toolchain
- Kokoro Core ML assets copied into `KokoroCoreMLTTS/Resources/Models`

## Project Layout

```
ios/
├── KokoroCoreMLTTS.xcodeproj     # Xcode workspace with single application target
├── KokoroCoreMLTTS/              # App sources, resources, entitlements, asset catalog
│   ├── Sources/                  # SwiftUI view + engine implementation
│   ├── Resources/Models          # Duration + decoder mlpackages bundled into the app
│   └── Resources/Voices          # Voice embeddings (float32 .bin files)
└── README.md                     # This file
```

## Getting Started

1. Open `ios/KokoroCoreMLTTS.xcodeproj` in Xcode.
2. Ensure `KokoroCoreMLTTS` is selected as the build target and choose an iOS 17 simulator or a connected device.
3. Build and run (`⌘R`). The app launches to a single SwiftUI screen with text input, voice picker, rate slider, and Speak/Stop controls.

### Core ML Assets

The project expects the following bundles in `Resources/Models`:

- `kokoro_duration.mlpackage`
- `kokoro_decoder_3s.mlpackage`
- `kokoro_decoder_10s.mlpackage`

The repo ships with the duration model and the decoder‑only 3 s/10 s models copied into this directory. If you have 45 s buckets or updated exports, drop them here and the runtime will detect them automatically.

### Voice Embeddings

Two example voices (`af_heart.bin`, `am_michael.bin`) are bundled for female and male baselines. Add more by copying `.bin` files into `Resources/Voices` and extending `VoiceLibrary.swift`.

### Phoneme Vocabulary

`Resources/phoneme_vocab.json` contains a small fallback mapping so the demo runs out of the box, but quality improves significantly when you replace it with the real `checkpoints/config.json` vocabulary. Copy the JSON into this directory and rename it to `phoneme_vocab.json` to enable the Kotlin‑accurate tokenizer.

## Architecture Overview

- **SwiftUI UI (`Sources/UI`)** – Single screen (`TTSView`) backed by `TTSViewModel`, providing text entry, voice selection, rate control, playback controls, and real‑time meters.
- **Engine Layer (`Sources/Engines`)** – `KokoroTTSEngine` orchestrates tokenisation, duration prediction, bucket selection, alignment, decoder invocation, audio rendering, and metrics logging. `AudioRenderer` handles streaming playback via `AVAudioEngine`. `SystemTTSEngine` wraps `AVSpeechSynthesizer` for A/B comparisons.
- **Model Wrappers (`Sources/Models`)** – Thin wrappers around the duration and decoder Core ML models manage lazy loading, batching, and tensor shape preparation.
- **Utilities (`Sources/Utilities`)** – Feature flags, metrics collector, multi‑array helpers, and voice library.

## Metrics & Instrumentation

The view model records:

- Cold/warm load timings for duration + decoder models
- Per segment inference time
- Total end‑to‑end latency
- Peak resident memory (approximate, RSS)
- Real‑time RMS level for a simple level meter

Metrics are surfaced in the UI and logged to the console for deeper inspection.

## Next Steps

- Replace the fallback phoneme map with the official vocabulary and tokenizer for production quality.
- Add additional decoder buckets (e.g. 45 s) once exported to Core ML.
- Integrate the Python tokenizer bridge or a native Swift G2P implementation for full Kokoro parity.
- Wire feature flags (`FeatureFlags`) to persist user defaults and enable developer toggles such as dumping WAV files.

## Troubleshooting

- **Missing models** – The runtime throws a user‑visible error if a `.mlpackage` cannot be found. Verify the file names and ensure they are added to the Copy Bundle Resources phase.
- **Silent output / artifacts** – Typically caused by placeholder phoneme IDs. Replace `phoneme_vocab.json` with the real vocabulary and verify the tokenizer output.
- **High latency on first run** – The app records cold start timings so you can measure warm cache performance. Use the metrics panel to compare.
- **System TTS comparison** – Toggle “Use System TTS” to route synthesis through `AVSpeechSynthesizer` for baseline behaviour.

## License

This sample inherits the repository’s license. Please ensure compliance with Kokoro’s model terms when redistributing the Core ML assets.
