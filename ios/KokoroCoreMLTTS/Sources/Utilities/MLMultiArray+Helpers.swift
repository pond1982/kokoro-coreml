import CoreML
import Foundation

extension MLMultiArray {
    static func makeFloatArray(shape: [NSNumber]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape, dataType: .float32)
    }

    static func from(vector: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: vector.count)], dataType: .float32)
        for (index, value) in vector.enumerated() {
            array[[0, NSNumber(value: index)]] = NSNumber(value: value)
        }
        return array
    }

    static func from3D(channels: Int, frames: Int, fill: (Int, Int) -> Float) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: channels), NSNumber(value: frames)], dataType: .float32)
        for channel in 0..<channels {
            for frame in 0..<frames {
                array[[0, NSNumber(value: channel), NSNumber(value: frame)]] = NSNumber(value: fill(channel, frame))
            }
        }
        return array
    }

    static func fromMatrix(_ matrix: [[Float]]) throws -> MLMultiArray {
        guard let columnCount = matrix.first?.count else {
            return try MLMultiArray(shape: [1, 0, 0], dataType: .float32)
        }
        return try from3D(channels: matrix.count, frames: columnCount) { row, col in
            matrix[row][col]
        }
    }

    func toFloatArray() -> [Float] {
        guard dataType == .float32 else {
            return (0..<count).map { Float(truncating: self[NSNumber(value: $0)]) }
        }
        let pointer = dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    func to2DArray(channels: Int, frames: Int) -> [[Float]] {
        var result = Array(repeating: Array(repeating: Float(0), count: frames), count: channels)
        for channel in 0..<channels {
            for frame in 0..<frames {
                let index = [NSNumber(value: 0), NSNumber(value: channel), NSNumber(value: frame)]
                result[channel][frame] = Float(truncating: self[index])
            }
        }
        return result
    }
}
