import AVFoundation
import CoreML
import Darwin
import Foundation

fileprivate func kokoroLog(_ items: Any..., function: String = #function) {
    let prefix = "[KokoroTTSEngine] \(function):"
    print(prefix, items.map { String(describing: $0) }.joined(separator: " "))
}

protocol KokoroTTSEngineDelegate: AnyObject {
    func engineDidStartSynthesis()
    func engineDidFinish(with metrics: InferenceMetrics)
    func engineDidUpdate(level: Float)
    func engineDidUpdate(elapsed: TimeInterval)
    func engineDidEncounter(error: Error)
}

final class KokoroTTSEngine: NSObject {
    weak var delegate: KokoroTTSEngineDelegate?

    let voiceLibrary = VoiceLibrary()
    private let durationModel = DurationPredictorML()
    private let decoder = DecoderML()
    private let audioRenderer = AudioRenderer()
    private let metrics = MetricsLogger()
    private let featureFlags: FeatureFlags
    private let queue = DispatchQueue(label: "kokoro.tts.engine")
    private var stopRequested = false
    private var elapsedTimer: Timer?
    private var synthesisStart: Date?
    private let textProcessor: TextProcessor
    private let fallbackProcessor = NaiveTextProcessor()
    private var durationModelWarmed = false
    private var warmedBuckets: Set<DecoderBucket> = []
    private let supportedBuckets: [DecoderBucket]

    override init() {
        self.featureFlags = .defaults
        if let processor = KokoroTextProcessor() {
            textProcessor = processor
        } else {
            textProcessor = NaiveTextProcessor()
        }
        let available = KokoroTTSEngine.detectBuckets()
        supportedBuckets = available.isEmpty ? [.threeSeconds] : available
        super.init()
        audioRenderer.delegate = self
    }

    func stop() {
        kokoroLog("Stop requested")
        queue.sync {
            stopRequested = true
            audioRenderer.stop()
            invalidateTimer()
        }
    }

    func speak(text: String, voice: VoiceDefinition, speed: Float) {
        kokoroLog("Speak start textLen=", text.count, "voice=", voice.id, "speed=", speed)
        stop()
        delegate?.engineDidStartSynthesis()
        stopRequested = false
        synthesisStart = Date()
        startElapsedTimer()

        queue.async {
            do {
                let warmStart = Date()
                try self.durationModel.warmUpIfNeeded()
                let durationWarm = Date().timeIntervalSince(warmStart)
                kokoroLog("Duration model warm done")
                if self.durationModelWarmed {
                    self.metrics.recordWarmLoad(durationWarm)
                } else {
                    self.metrics.recordColdLoad(durationWarm)
                    self.durationModelWarmed = true
                }

                let decoderBuckets = self.supportedBuckets.filter { !self.warmedBuckets.contains($0) }
                if !decoderBuckets.isEmpty {
                    let bucketWarmStart = Date()
                    try self.decoder.warmUp(buckets: decoderBuckets)
                    let warm = Date().timeIntervalSince(bucketWarmStart)
                    if self.warmedBuckets.isEmpty {
                        self.metrics.recordColdLoad(warm)
                    } else {
                        self.metrics.recordWarmLoad(warm)
                    }
                    decoderBuckets.forEach { self.warmedBuckets.insert($0) }
                    kokoroLog("Decoder warm done buckets=", decoderBuckets)
                }

                let metricsStart = Date()
                let voiceEmbedding = try self.voiceLibrary.loadEmbedding(for: voice)
                kokoroLog("Loaded voice embedding count=", voiceEmbedding.count)
                let refS = try self.makeRefSArray(from: voiceEmbedding)
                let sentences: [TokenizedSentence]
                 do {
                     sentences = try self.textProcessor.tokenize(text)
                     kokoroLog("Tokenized with primary processor. sentences=", sentences.count, "tokenCounts=", sentences.map { $0.tokens.count })
                 } catch {
                     sentences = try self.fallbackProcessor.tokenize(text)
                     kokoroLog("Tokenized with fallback processor. sentences=", sentences.count, "tokenCounts=", sentences.map { $0.tokens.count })
                 }
                kokoroLog("Begin synthesis segments count=", sentences.count)
                var generatedAudio: [Float] = []

                for sentence in sentences {
                    kokoroLog("Segment start idx=", generatedAudio.count, "tokens=", sentence.tokens.count)
                    if self.stopRequested { return }
                    let bucket = self.selectBucket(for: sentence.tokens.count, speed: speed)
                    kokoroLog("Selected bucket=", bucket, "tokenCapacity=", bucket.tokenCapacity)
                    let segmentStart = Date()
                    let result = try self.runPipeline(
                        tokens: sentence.tokens,
                        refS: refS,
                        bucket: bucket,
                        speed: speed
                    )
                    generatedAudio.append(contentsOf: result)
                    if let minAmp = result.min(), let maxAmp = result.max() {
                        kokoroLog("Segment audio samples=", result.count, "range=", String(format: "%.6f", minAmp), "to", String(format: "%.6f", maxAmp))
                    } else {
                        kokoroLog("Segment audio samples=", result.count)
                    }
                    let delta = Date().timeIntervalSince(segmentStart)
                    self.metrics.recordSegment(duration: delta)
                }

                if self.stopRequested { return }
                let total = Date().timeIntervalSince(metricsStart)
                self.metrics.recordTotal(total)
                self.metrics.recordPeakMemory(self.captureMemory())
                self.invalidateTimer()
                kokoroLog("Synthesis complete totalSamples=", generatedAudio.count)

                do {
                    try self.audioRenderer.play(waveform: generatedAudio)
                    kokoroLog("Playback started")
                } catch {
                    kokoroLog("Playback error:", error.localizedDescription)
                    throw error
                }

                if self.featureFlags.dumpAudioToFiles {
                    try self.dumpWaveform(generatedAudio)
                }
            } catch {
                DispatchQueue.main.async {
                    self.delegate?.engineDidEncounter(error: error)
                    self.invalidateTimer()
                }
            }
        }
    }

    private func runPipeline(tokens: [Int32], refS: MLMultiArray, bucket: DecoderBucket, speed: Float) throws -> [Float] {
        kokoroLog("runPipeline tokens=", tokens.count, "bucket=", bucket)
        let prediction = try durationModel.predict(tokenIDs: tokens, refS: refS, speed: speed)

        let tokenCount = min(tokens.count + 2, prediction.predDurations.count)
        let durations = Array(prediction.predDurations.prefix(tokenCount))
        let alignment = buildAlignmentMatrix(durations: durations, frameCount: bucket.frameCount)
        kokoroLog("Durations tokenCount=", tokenCount, "frameCount=", bucket.frameCount)

        let tEnChannels = Int(truncating: prediction.tEn.shape[1])
        let tEnLength = Int(truncating: prediction.tEn.shape[2])
        let tEnMatrix = prediction.tEn.to2DArray(channels: tEnChannels, frames: tEnLength)
        kokoroLog("ASR matrix dims=", tEnChannels, "x", bucket.frameCount)
        let asrMatrix = multiply(features: tEnMatrix, alignment: alignment)

        let f0Noise = generateF0Noise(for: bucket, alignment: alignment)
        let asrArray = try MLMultiArray.fromMatrix(asrMatrix)
        let f0Array = try MLMultiArray.from(vector: f0Noise.f0)
        let noiseArray = try MLMultiArray.from(vector: f0Noise.noise)
        kokoroLog("Decoder input shapes asr=", asrArray.shape, "f0=", f0Array.shape, "noise=", noiseArray.shape)

        let waveform = try decoder.decode(bucket: bucket, asr: asrArray, f0: f0Array, noise: noiseArray, refS: refS)
        return waveform.toFloatArray()
    }

    private func makeRefSArray(from embedding: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 256], dataType: .float32)
        for (index, value) in embedding.prefix(256).enumerated() {
            array[[0, NSNumber(value: index)]] = NSNumber(value: value)
        }
        kokoroLog("refS array created shape=", array.shape)
        return array
    }

    private func buildAlignmentMatrix(durations: [Int], frameCount: Int) -> [[Float]] {
        var matrix = Array(repeating: Array(repeating: Float(0), count: frameCount), count: durations.count)
        let total = max(durations.reduce(0, +), 1)
        let scale = Float(frameCount) / Float(total)
        var cursor = 0
        for (index, duration) in durations.enumerated() {
            let scaled = max(1, Int(round(Float(duration) * scale)))
            let limit = min(frameCount, cursor + scaled)
            for frame in cursor..<limit {
                matrix[index][frame] = 1
            }
            cursor = limit
            if cursor >= frameCount { break }
        }
        if cursor < frameCount {
            for frame in cursor..<frameCount {
                matrix[max(durations.count - 1, 0)][frame] = 1
            }
        }
        return matrix
    }

    private func multiply(features: [[Float]], alignment: [[Float]]) -> [[Float]] {
        let channels = features.count
        let tokens = features.first?.count ?? 0
        let frames = alignment.first?.count ?? 0
        var output = Array(repeating: Array(repeating: Float(0), count: frames), count: channels)
        for channel in 0..<channels {
            for frame in 0..<frames {
                var sum: Float = 0
                for token in 0..<min(tokens, alignment.count) {
                    sum += features[channel][token] * alignment[token][frame]
                }
                output[channel][frame] = sum
            }
        }
        return output
    }

    private func generateF0Noise(for bucket: DecoderBucket, alignment: [[Float]]) -> (f0: [Float], noise: [Float]) {
        let frames = bucket.f0Count
        var f0 = [Float](repeating: 0, count: frames)
        var noise = [Float](repeating: 0.05, count: frames)
        for frame in 0..<frames {
            let progress = Float(frame) / Float(max(frames - 1, 1))
            f0[frame] = 0.4 + 0.3 * sin(progress * 6.28318)
            noise[frame] = 0.05 + 0.02 * cos(progress * 3.14159)
        }
        return (f0, noise)
    }

    private func selectBucket(for tokenCount: Int, speed: Float) -> DecoderBucket {
        let sorted = supportedBuckets.sorted { $0.duration < $1.duration }
        if sorted.isEmpty {
            kokoroLog("selectBucket fallback chose=", DecoderBucket.threeSeconds, "for adjustedTokens=0")
            return .threeSeconds
        }
        let adjustedTokens = Int(Float(tokenCount) / max(speed, 0.5))
        for bucket in sorted {
            if adjustedTokens <= bucket.tokenCapacity {
                kokoroLog("selectBucket chose=", bucket, "for adjustedTokens=", adjustedTokens)
                return bucket
            }
        }
        kokoroLog("selectBucket fallback chose=", sorted.last ?? .threeSeconds, "for adjustedTokens=", adjustedTokens)
        return sorted.last ?? .threeSeconds
    }

    private func captureMemory() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / 1_048_576.0
    }

    private func dumpWaveform(_ audio: [Float]) throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("kokoro-\(UUID().uuidString).raw")
        let data = Data(bytes: audio, count: audio.count * MemoryLayout<Float>.size)
        try data.write(to: url)
    }

    private func startElapsedTimer() {
        DispatchQueue.main.async {
            self.elapsedTimer?.invalidate()
            self.elapsedTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                guard let self, let start = self.synthesisStart else { return }
                self.delegate?.engineDidUpdate(elapsed: Date().timeIntervalSince(start))
            }
        }
    }

    private func invalidateTimer() {
        DispatchQueue.main.async {
            self.elapsedTimer?.invalidate()
            self.elapsedTimer = nil
        }
    }

    private static func detectBuckets(bundle: Bundle = .main) -> [DecoderBucket] {
        let detected = DecoderBucket.allCases.filter { bucket in
            bundle.url(forResource: bucket.modelName, withExtension: "mlmodelc") != nil ||
            bundle.url(forResource: bucket.modelName, withExtension: "mlpackage") != nil
        }
        print("[KokoroTTSEngine detectBuckets]", detected)
        return detected
    }
}

extension KokoroTTSEngine: AudioRendererDelegate {
    func audioRendererDidFinishPlaying() {
        let snapshot = metrics.flush()
        DispatchQueue.main.async {
            self.delegate?.engineDidFinish(with: snapshot)
        }
    }

    func audioRenderer(levelDidUpdate rms: Float) {
        DispatchQueue.main.async {
            self.delegate?.engineDidUpdate(level: rms)
        }
    }
}
