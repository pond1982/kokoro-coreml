import Foundation

struct InferenceMetrics: Identifiable {
    let id = UUID()
    var coldLoadDuration: TimeInterval?
    var warmLoadDuration: TimeInterval?
    var segmentDurations: [TimeInterval]
    var totalDuration: TimeInterval?
    var peakMemoryMB: Double?
    var timestamp: Date

    init(
        coldLoadDuration: TimeInterval? = nil,
        warmLoadDuration: TimeInterval? = nil,
        segmentDurations: [TimeInterval] = [],
        totalDuration: TimeInterval? = nil,
        peakMemoryMB: Double? = nil,
        timestamp: Date = .now
    ) {
        self.coldLoadDuration = coldLoadDuration
        self.warmLoadDuration = warmLoadDuration
        self.segmentDurations = segmentDurations
        self.totalDuration = totalDuration
        self.peakMemoryMB = peakMemoryMB
        self.timestamp = timestamp
    }
}

/// Simple metrics collector that keeps timing information per synthesis session.
final class MetricsLogger: @unchecked Sendable {
    private let queue = DispatchQueue(label: "tts.metrics.logger", qos: .utility)
    private var activeMetrics = InferenceMetrics()

    func recordColdLoad(_ interval: TimeInterval) {
        queue.async { self.activeMetrics.coldLoadDuration = interval }
    }

    func recordWarmLoad(_ interval: TimeInterval) {
        queue.async { self.activeMetrics.warmLoadDuration = interval }
    }

    func recordSegment(duration: TimeInterval) {
        queue.async { self.activeMetrics.segmentDurations.append(duration) }
    }

    func recordTotal(_ interval: TimeInterval) {
        queue.async { self.activeMetrics.totalDuration = interval }
    }

    func recordPeakMemory(_ megabytes: Double) {
        queue.async { self.activeMetrics.peakMemoryMB = megabytes }
    }

    func flush() -> InferenceMetrics {
        queue.sync {
            let snapshot = activeMetrics
            activeMetrics = InferenceMetrics()
            return snapshot
        }
    }
}
