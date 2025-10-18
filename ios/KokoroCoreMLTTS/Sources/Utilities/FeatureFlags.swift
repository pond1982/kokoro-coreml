import Foundation

/// Runtime-configurable feature flags used to toggle optional behaviours.
struct FeatureFlags {
    var useSystemTTS: Bool
    var dumpAudioToFiles: Bool
    var enableWaveformMeter: Bool

    static let defaults = FeatureFlags(
        useSystemTTS: false,
        dumpAudioToFiles: false,
        enableWaveformMeter: true
    )
}
