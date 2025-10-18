import AVFoundation
import Foundation

final class SystemTTSEngine: NSObject {
    private let synthesizer = AVSpeechSynthesizer()
    var onDidFinish: (() -> Void)?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func speak(text: String, voiceIdentifier: String?, rate: Float) {
        stop()
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = rate
        if let voiceIdentifier {
            utterance.voice = AVSpeechSynthesisVoice(identifier: voiceIdentifier)
        }
        synthesizer.speak(utterance)
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}

extension SystemTTSEngine: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        onDidFinish?()
    }
}
