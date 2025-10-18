import AVFoundation
import Foundation

protocol AudioRendererDelegate: AnyObject {
    func audioRendererDidFinishPlaying()
    func audioRenderer(levelDidUpdate rms: Float)
}

final class AudioRenderer: NSObject {
    weak var delegate: AudioRendererDelegate?

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var meterTimer: CADisplayLink?
    private let meterQueue = DispatchQueue(label: "tts.audio.meter")
    private var currentBuffer: AVAudioPCMBuffer?

    override init() {
        super.init()
        engine.attach(player)
        let format = AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)!
        engine.connect(player, to: engine.mainMixerNode, format: format)
        engine.mainMixerNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.meterQueue.async {
                guard let channelData = buffer.floatChannelData?.pointee else { return }
                let frameLength = Int(buffer.frameLength)
                let rms: Float
                if frameLength > 0 {
                    var sum: Float = 0
                    for index in 0..<frameLength {
                        let sample = channelData[index]
                        sum += sample * sample
                    }
                    rms = sqrt(sum / Float(frameLength))
                } else {
                    rms = 0
                }
                DispatchQueue.main.async {
                    self?.delegate?.audioRenderer(levelDidUpdate: rms)
                }
            }
        }
        NotificationCenter.default.addObserver(self, selector: #selector(handleEngineConfigChange), name: .AVAudioEngineConfigurationChange, object: engine)
    }

    deinit {
        engine.mainMixerNode.removeTap(onBus: 0)
        NotificationCenter.default.removeObserver(self)
    }

    func play(waveform: [Float], sampleRate: Double = 24_000) throws {
        try startEngineIfNeeded(sampleRate: sampleRate)
        guard let format = player.inputFormat(forBus: 0).withSampleRate(sampleRate) else { return }

        let frameCount = AVAudioFrameCount(waveform.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return }
        waveform.withUnsafeBufferPointer { pointer in
            guard let base = pointer.baseAddress else { return }
            buffer.frameLength = frameCount
            memcpy(buffer.floatChannelData?.pointee, base, Int(frameCount) * MemoryLayout<Float>.size)
        }

        player.scheduleBuffer(buffer, at: nil, options: []) { [weak self] in
            DispatchQueue.main.async {
                self?.delegate?.audioRendererDidFinishPlaying()
            }
        }
        currentBuffer = buffer
        player.play()
    }

    func stop() {
        player.stop()
        engine.stop()
        currentBuffer = nil
    }

    private func startEngineIfNeeded(sampleRate: Double) throws {
        guard !engine.isRunning else { return }
        try AVAudioSession.sharedInstance().setCategory(.playback, options: [.mixWithOthers])
        try AVAudioSession.sharedInstance().setActive(true)
        if engine.outputNode.outputFormat(forBus: 0).sampleRate != sampleRate {
            // Reconnect output format if sample rate mismatch occurs.
            engine.disconnectNodeOutput(player)
            let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
            engine.connect(player, to: engine.mainMixerNode, format: format)
        }
        try engine.start()
    }

    @objc private func handleEngineConfigChange(_ notification: Notification) {
        do {
            if let buffer = currentBuffer {
                if let channelPointer = buffer.floatChannelData?.pointee {
                    let existing = Array(UnsafeBufferPointer(start: channelPointer, count: Int(buffer.frameLength)))
                    try play(waveform: existing)
                }
            }
        } catch {
            print("Failed to restart audio engine: \(error)")
        }
    }
}

private extension AVAudioFormat {
    func withSampleRate(_ rate: Double) -> AVAudioFormat? {
        AVAudioFormat(commonFormat: commonFormat, sampleRate: rate, channels: channelCount, interleaved: isInterleaved)
    }
}
