import Combine
import AVFoundation
import Foundation

@MainActor
final class TTSViewModel: ObservableObject {
    @Published var inputText: String = "Hello Kokoro!"
    @Published var selectedVoiceIndex: Int = 0
    @Published var speechRate: Double = 1.0
    @Published var isSpeaking: Bool = false
    @Published var elapsed: TimeInterval = 0
    @Published var level: Float = 0
    @Published var metricsHistory: [InferenceMetrics] = []
    @Published var useSystemTTS: Bool = false

    let voices: [VoiceDefinition]
    private let kokoroEngine: KokoroTTSEngine
    private let systemEngine = SystemTTSEngine()
    private var cancellables = Set<AnyCancellable>()

    init() {
        let library = VoiceLibrary()
        voices = library.voices
        kokoroEngine = KokoroTTSEngine()
        kokoroEngine.delegate = self
        systemEngine.onDidFinish = { [weak self] in
            Task { @MainActor in self?.isSpeaking = false }
        }
    }

    func speak() {
        guard !inputText.isEmpty else { return }
        stop()
        isSpeaking = true
        if useSystemTTS {
            let voice = voices[selectedVoiceIndex]
            systemEngine.speak(text: inputText, voiceIdentifier: nil, rate: speechRateToSystemRate())
        } else {
            let voice = voices[selectedVoiceIndex]
            kokoroEngine.speak(text: inputText, voice: voice, speed: Float(speechRate))
        }
    }

    func stop() {
        if useSystemTTS {
            systemEngine.stop()
        } else {
            kokoroEngine.stop()
        }
        isSpeaking = false
    }

    func resetMetrics() {
        metricsHistory.removeAll()
    }

    private func speechRateToSystemRate() -> Float {
        let normalized = (speechRate - 0.75) / (1.25 - 0.75)
        let mapped = 0.4 + normalized * Double(AVSpeechUtteranceDefaultSpeechRate - 0.4)
        return Float(mapped)
    }
}

extension TTSViewModel: KokoroTTSEngineDelegate {
    func engineDidStartSynthesis() {
        Task { @MainActor in
            isSpeaking = true
            elapsed = 0
        }
    }

    func engineDidFinish(with metrics: InferenceMetrics) {
        Task { @MainActor in
            isSpeaking = false
            metricsHistory.append(metrics)
        }
    }

    func engineDidUpdate(level: Float) {
        Task { @MainActor in
            self.level = level
        }
    }

    func engineDidUpdate(elapsed: TimeInterval) {
        Task { @MainActor in
            self.elapsed = elapsed
        }
    }

    func engineDidEncounter(error: Error) {
        Task { @MainActor in
            isSpeaking = false
            print("Engine error: \(error)")
        }
    }
}
