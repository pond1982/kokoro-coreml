import Foundation

struct TokenizedSentence {
    var tokens: [Int32]
    var phonemes: [String]
}

protocol TextProcessor {
    func tokenize(_ text: String) throws -> [TokenizedSentence]
}

enum TextProcessingError: LocalizedError {
    case missingVocabulary
    case unsupportedCharacter(Character)
    case emptyResult

    var errorDescription: String? {
        switch self {
        case .missingVocabulary:
            return "Missing phoneme vocabulary. Add checkpoints/config.json or phoneme_vocab.json to the app bundle."
        case .unsupportedCharacter(let character):
            return "Encountered unsupported character '\(character)'."
        case .emptyResult:
            return "No phonemes generated for the provided text."
        }
    }
}

/// Loads Kokoro's phoneme vocabulary from a JSON file and creates a minimal tokenizer.
final class KokoroTextProcessor: TextProcessor {
    private let vocab: [String: Int]
    private let maxTokens = 128

    init?(bundle: Bundle = .main) {
        if let url = bundle.url(forResource: "phoneme_vocab", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let decoded = try? JSONDecoder().decode([String: Int].self, from: data) {
            vocab = decoded
        } else if let url = bundle.url(forResource: "config", withExtension: "json"),
                  let data = try? Data(contentsOf: url),
                  let decoded = try? JSONDecoder().decode(ConfigWrapper.self, from: data) {
            vocab = decoded.vocab
        } else {
            vocab = [:]
            return nil
        }
    }

    func tokenize(_ text: String) throws -> [TokenizedSentence] {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { throw TextProcessingError.emptyResult }

        let sentences = KokoroTextProcessor.segment(cleaned)
        let result: [TokenizedSentence] = try sentences.compactMap { sentence in
            let phonemes = Self.simplePhonemize(sentence)
            let ids = try phonemes.compactMap { symbol -> Int32 in
                guard let id = vocab[symbol] else {
                    throw TextProcessingError.unsupportedCharacter(Character(symbol))
                }
                return Int32(id)
            }

            guard !ids.isEmpty else { throw TextProcessingError.emptyResult }
            let truncated = Array(ids.prefix(maxTokens - 2)) // allow BOS/EOS padding later
            return TokenizedSentence(tokens: truncated, phonemes: phonemes)
        }

        guard !result.isEmpty else { throw TextProcessingError.emptyResult }
        return result
    }

    private static func segment(_ text: String) -> [String] {
        var sentences: [String] = []
        var current = ""
        for character in text {
            current.append(character)
            if ".!?;:".contains(character) {
                sentences.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current.removeAll()
            }
        }
        if !current.isEmpty {
            sentences.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return sentences
    }

    private static func simplePhonemize(_ text: String) -> [String] {
        let lower = text.lowercased()
        var phonemes: [String] = []
        for character in lower {
            if let symbol = fallbackCharacterToPhoneme[character] {
                phonemes.append(symbol)
            } else if character.isWhitespace {
                phonemes.append("_")
            } else {
                phonemes.append(String(character))
            }
        }
        return phonemes
    }

    private static let fallbackCharacterToPhoneme: [Character: String] = [
        "a": "a",
        "b": "b",
        "c": "k",
        "d": "d",
        "e": "e",
        "f": "f",
        "g": "g",
        "h": "h",
        "i": "i",
        "j": "dʒ",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "o": "oʊ",
        "p": "p",
        "q": "k",
        "r": "ɹ",
        "s": "s",
        "t": "t",
        "u": "u",
        "v": "v",
        "w": "w",
        "x": "ks",
        "y": "j",
        "z": "z",
        "'": "ˈ",
        ",": ",",
        ".": ".",
        "?": "?",
        "!": "!",
        ":": ":",
        ";": ";"
    ]

    private struct ConfigWrapper: Decodable {
        let vocab: [String: Int]
    }
}

/// Extremely small fallback tokenizer used when no vocab is provided.
final class NaiveTextProcessor: TextProcessor {
    func tokenize(_ text: String) throws -> [TokenizedSentence] {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { throw TextProcessingError.emptyResult }

        let ids = cleaned.unicodeScalars.map { Int32($0.value % 255) }
        return [TokenizedSentence(tokens: ids, phonemes: cleaned.map { String($0) })]
    }
}
