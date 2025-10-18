import CoreML
import Foundation

struct DurationPredictionOutput {
    var predDurations: [Int]
    var d: MLMultiArray
    var tEn: MLMultiArray
    var s: MLMultiArray
    var refSOut: MLMultiArray
}

enum DurationPredictorError: LocalizedError {
    case modelNotFound
    case predictionFailure(Error)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Unable to locate kokoro_duration.mlpackage in the app bundle."
        case .predictionFailure(let error):
            return "Duration prediction failed: \(error.localizedDescription)"
        }
    }
}

final class DurationPredictorML {
    private static let modelName = "kokoro_duration"
    private lazy var configuration: MLModelConfiguration = {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        config.allowLowPrecisionAccumulationOnGPU = true
        return config
    }()

    private var model: MLModel?
    private let lock = NSLock()

    func warmUpIfNeeded() throws {
        _ = try loadModelIfNeeded()
    }

    func predict(
        tokenIDs: [Int32],
        refS: MLMultiArray,
        speed: Float
    ) throws -> DurationPredictionOutput {
        let model = try loadModelIfNeeded()

        let (padded, attentionMask) = DurationPredictorML.pad(tokens: tokenIDs, to: 128)

        let inputFeatures = DurationInput(
            inputIDs: try MLMultiArray(fromInt32: padded),
            attentionMask: try MLMultiArray(fromInt32: attentionMask),
            refS: refS,
            speed: try MLMultiArray(from: [speed])
        )

        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: inputFeatures)
        } catch {
            throw DurationPredictorError.predictionFailure(error)
        }

        guard
            let predDur = output.featureValue(for: "pred_dur")?.multiArrayValue,
            let d = output.featureValue(for: "d")?.multiArrayValue,
            let tEn = output.featureValue(for: "t_en")?.multiArrayValue,
            let s = output.featureValue(for: "s")?.multiArrayValue,
            let refSOut = output.featureValue(for: "ref_s_out")?.multiArrayValue
        else {
            throw DurationPredictorError.predictionFailure(
                NSError(domain: "kokoro.duration", code: -1, userInfo: [NSLocalizedDescriptionKey: "Missing outputs"])
            )
        }

        let durations = predDur.asIntArray()
        return DurationPredictionOutput(
            predDurations: durations,
            d: d,
            tEn: tEn,
            s: s,
            refSOut: refSOut
        )
    }

    private func loadModelIfNeeded() throws -> MLModel {
        lock.lock()
        defer { lock.unlock() }

        if let model {
            return model
        }

        guard let url = Bundle.main.url(forResource: Self.modelName, withExtension: "mlmodelc") ??
                Bundle.main.url(forResource: Self.modelName, withExtension: "mlpackage")
        else {
            throw DurationPredictorError.modelNotFound
        }

        let compiledURL: URL
        if url.pathExtension == "mlpackage" {
            compiledURL = try MLModel.compileModel(at: url)
        } else {
            compiledURL = url
        }

        let loaded = try MLModel(contentsOf: compiledURL, configuration: configuration)
        model = loaded
        return loaded
    }

    private static func pad(tokens: [Int32], to length: Int) -> ([Int32], [Int32]) {
        let bos: Int32 = 0
        let eos: Int32 = 0
        let truncated = tokens.prefix(length - 2)
        var input = [Int32](repeating: 0, count: length)
        var mask = [Int32](repeating: 0, count: length)
        guard let firstIndex = input.indices.first else { return (input, mask) }
        input[firstIndex] = bos
        mask[firstIndex] = 1
        for (offset, value) in truncated.enumerated() {
            let index = firstIndex + offset + 1
            input[index] = value
            mask[index] = 1
        }
        let eosIndex = firstIndex + truncated.count + 1
        if eosIndex < length {
            input[eosIndex] = eos
            mask[eosIndex] = 1
        }
        return (input, mask)
    }
}

private final class DurationInput: MLFeatureProvider {
    var inputIDs: MLMultiArray
    var attentionMask: MLMultiArray
    var refS: MLMultiArray
    var speed: MLMultiArray

    init(inputIDs: MLMultiArray, attentionMask: MLMultiArray, refS: MLMultiArray, speed: MLMultiArray) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
        self.refS = refS
        self.speed = speed
    }

    var featureNames: Set<String> { ["input_ids", "attention_mask", "ref_s", "speed"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "input_ids": return MLFeatureValue(multiArray: inputIDs)
        case "attention_mask": return MLFeatureValue(multiArray: attentionMask)
        case "ref_s": return MLFeatureValue(multiArray: refS)
        case "speed": return MLFeatureValue(multiArray: speed)
        default: return nil
        }
    }
}

private extension MLMultiArray {
    convenience init(fromInt32 values: [Int32]) throws {
        try self.init(shape: [1, NSNumber(value: values.count)], dataType: .int32)
        let row = 0 as NSNumber
        for (index, value) in values.enumerated() {
            self[[row, NSNumber(value: index)]] = NSNumber(value: value)
        }
    }

    convenience init(from values: [Float]) throws {
        try self.init(shape: [NSNumber(value: values.count)], dataType: .float32)
        for (index, value) in values.enumerated() {
            self[[NSNumber(value: index)]] = NSNumber(value: value)
        }
    }

    func asIntArray() -> [Int] {
        var result: [Int] = []
        result.reserveCapacity(count)
        // Handle 1D arrays
        if shape.count == 1 {
            let n = Int(truncating: shape[0])
            for i in 0..<n {
                result.append(Int(truncating: self[[NSNumber(value: i)]]))
            }
            return result
        }
        // Handle 2D arrays (e.g., [1, N])
        if shape.count == 2 {
            let rows = Int(truncating: shape[0])
            let cols = Int(truncating: shape[1])
            for r in 0..<rows {
                for c in 0..<cols {
                    result.append(Int(truncating: self[[NSNumber(value: r), NSNumber(value: c)]]))
                }
            }
            return result
        }
        // Fallback: linear read
        for i in 0..<count {
            result.append(Int(truncating: self[[NSNumber(value: i)]]))
        }
        return result
    }
}
