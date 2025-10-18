import Foundation

struct VoiceDefinition: Identifiable, Hashable {
    let id: String
    let displayName: String
    let assetName: String
    let gender: String
    let language: String
}

enum VoiceLibraryError: LocalizedError {
    case missingVoiceAsset(String)
    case invalidVoiceData(String)

    var errorDescription: String? {
        switch self {
        case .missingVoiceAsset(let voice):
            return "Voice asset \(voice) is missing from the bundle."
        case .invalidVoiceData(let voice):
            return "Voice asset \(voice) is invalid."
        }
    }
}

final class VoiceLibrary {
    let voices: [VoiceDefinition] = [
        VoiceDefinition(id: "af_heart", displayName: "Heart", assetName: "af_heart", gender: "Female", language: "en-US"),
        VoiceDefinition(id: "am_michael", displayName: "Michael", assetName: "am_michael", gender: "Male", language: "en-US")
    ]

    func loadEmbedding(for voice: VoiceDefinition, bundle: Bundle = .main) throws -> [Float] {
        guard let url = bundle.url(forResource: voice.assetName, withExtension: "bin") else {
            throw VoiceLibraryError.missingVoiceAsset(voice.assetName)
        }
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.size
        var result = [Float](repeating: 0, count: count)
        _ = result.withUnsafeMutableBytes { resultPtr in
            data.copyBytes(to: resultPtr)
        }
        return result
    }
}
