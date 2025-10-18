import CoreML
import Foundation

fileprivate func decoderLog(_ items: Any..., function: String = #function) {
    let prefix = "[DecoderML] \(function):"
    print(prefix, items.map { String(describing: $0) }.joined(separator: " "))
}

enum DecoderBucket: String, CaseIterable {
    case threeSeconds
    case tenSeconds
    case fortyFiveSeconds

    var modelName: String {
        switch self {
        case .threeSeconds: return "kokoro_decoder_3s"
        case .tenSeconds: return "kokoro_decoder_10s"
        case .fortyFiveSeconds: return "kokoro_decoder_45s"
        }
    }

    var duration: TimeInterval {
        switch self {
        case .threeSeconds: return 3.0
        case .tenSeconds: return 10.0
        case .fortyFiveSeconds: return 45.0
        }
    }

    var frameCount: Int {
        switch self {
        case .threeSeconds: return 120
        case .tenSeconds: return 400
        case .fortyFiveSeconds: return 1800
        }
    }

    var waveformSamples: Int {
        switch self {
        case .threeSeconds: return 3 * 24_000
        case .tenSeconds: return 10 * 24_000
        case .fortyFiveSeconds: return 45 * 24_000
        }
    }

    var f0Count: Int { frameCount * 2 }

    var tokenCapacity: Int {
        switch self {
        case .threeSeconds: return 64
        case .tenSeconds: return 220
        case .fortyFiveSeconds: return 512
        }
    }
}

enum DecoderError: LocalizedError {
    case modelNotFound(String)
    case predictionFailure(Error)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Decoder model \(name) could not be located in the bundle."
        case .predictionFailure(let error):
            return "Decoder prediction failed: \(error.localizedDescription)"
        }
    }
}

final class DecoderML {
    private var cachedModels: [DecoderBucket: MLModel] = [:]
    private let lock = NSLock()

    private lazy var configuration: MLModelConfiguration = {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        config.allowLowPrecisionAccumulationOnGPU = true
        return config
    }()

    func warmUp(buckets: [DecoderBucket]) throws {
        for bucket in buckets {
            decoderLog("Warming bucket", bucket, "modelName=", bucket.modelName)
            _ = try loadModel(for: bucket)
        }
    }

    func decode(
        bucket: DecoderBucket,
        asr: MLMultiArray,
        f0: MLMultiArray,
        noise: MLMultiArray,
        refS: MLMultiArray
    ) throws -> MLMultiArray {
        decoderLog("Decode start bucket=", bucket, "frames=", bucket.frameCount, "f0Count=", bucket.f0Count)
        let model = try loadModel(for: bucket)
        decoderLog("Model loaded for bucket=", bucket)
        decoderLog("Input shapes asr=", asr.shape, "f0=", f0.shape, "noise=", noise.shape, "refS=", refS.shape)
        let input = DecoderInput(asr: asr, f0: f0, noise: noise, refS: refS)
        let output: MLFeatureProvider
        let t0 = Date()
        do {
            output = try model.prediction(from: input)
            let dt = Date().timeIntervalSince(t0)
            decoderLog("Prediction ok in", String(format: "%.3f", dt), "s")
        } catch {
            decoderLog("Prediction failed:", error.localizedDescription)
            throw DecoderError.predictionFailure(error)
        }

        guard let waveform = output.featureValue(for: "waveform")?.multiArrayValue else {
            throw DecoderError.predictionFailure(
                NSError(domain: "kokoro.decoder", code: -1, userInfo: [NSLocalizedDescriptionKey: "Missing waveform output"])
            )
        }

        let arr = waveform.toFloatArray()
        if let minAmp = arr.min(), let maxAmp = arr.max() {
            decoderLog("Waveform samples=", arr.count, "range=", String(format: "%.6f", minAmp), "to", String(format: "%.6f", maxAmp))
        } else {
            decoderLog("Waveform samples=", arr.count)
        }
        return waveform
    }

    private func loadModel(for bucket: DecoderBucket) throws -> MLModel {
        lock.lock()
        defer { lock.unlock() }
        decoderLog("Load model for bucket=", bucket)

        if let cached = cachedModels[bucket] {
            decoderLog("Using cached model for bucket=", bucket)
            return cached
        }

        let name = bucket.modelName
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") ??
                Bundle.main.url(forResource: name, withExtension: "mlpackage")
        else {
            throw DecoderError.modelNotFound(name)
        }
        decoderLog("Found model resource:", url.lastPathComponent)

        let compiledURL: URL
        if url.pathExtension == "mlpackage" {
            compiledURL = try MLModel.compileModel(at: url)
        } else {
            compiledURL = url
        }
        decoderLog("Compiled/ready model at:", compiledURL.lastPathComponent)

        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        decoderLog("Model loaded successfully for bucket=", bucket)
        cachedModels[bucket] = model
        return model
    }
}

private final class DecoderInput: MLFeatureProvider {
    var asr: MLMultiArray
    var f0: MLMultiArray
    var noise: MLMultiArray
    var refS: MLMultiArray

    init(asr: MLMultiArray, f0: MLMultiArray, noise: MLMultiArray, refS: MLMultiArray) {
        self.asr = asr
        self.f0 = f0
        self.noise = noise
        self.refS = refS
        decoderLog("DecoderInput init asr=", asr.shape, "f0=", f0.shape, "noise=", noise.shape, "refS=", refS.shape)
    }

    var featureNames: Set<String> { ["asr", "F0_pred", "N_pred", "ref_s"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "asr": return MLFeatureValue(multiArray: asr)
        case "F0_pred": return MLFeatureValue(multiArray: f0)
        case "N_pred": return MLFeatureValue(multiArray: noise)
        case "ref_s": return MLFeatureValue(multiArray: refS)
        default:
            decoderLog("Requested unknown feature:", featureName)
            return nil
        }
    }
}
