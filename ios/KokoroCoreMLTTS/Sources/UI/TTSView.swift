import SwiftUI

struct TTSView: View {
    @StateObject private var viewModel = TTSViewModel()
    @FocusState private var isTextFocused: Bool

    var body: some View {
        VStack(spacing: 16) {
            textInput
            controlSection
            progressSection
            metricsSection
        }
        .padding()
        .navigationTitle("Kokoro CoreML TTS")
        .toolbar { ToolbarItemGroup(placement: .keyboard) { keyboardToolbar } }
    }

    private var textInput: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Input Text")
                .font(.headline)
            TextEditor(text: $viewModel.inputText)
                .frame(minHeight: 120)
                .focused($isTextFocused)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray.opacity(0.2)))
            HStack {
                Spacer()
                Text("\(viewModel.inputText.count)/500")
                    .font(.caption)
                    .foregroundStyle(viewModel.inputText.count > 500 ? .red : .secondary)
            }
        }
    }

    private var controlSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker("Voice", selection: $viewModel.selectedVoiceIndex) {
                ForEach(Array(viewModel.voices.enumerated()), id: \.offset) { entry in
                    Text(entry.element.displayName).tag(entry.offset)
                }
            }
            .pickerStyle(.menu)

            VStack(alignment: .leading) {
                Text("Rate: \(String(format: "%.2fx", viewModel.speechRate))")
                Slider(value: $viewModel.speechRate, in: 0.75...1.25, step: 0.05)
            }

            Toggle("Use System TTS", isOn: $viewModel.useSystemTTS)

            HStack(spacing: 16) {
                Button(action: viewModel.speak) {
                    Label("Speak", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isSpeaking && !viewModel.useSystemTTS)

                Button(action: viewModel.stop) {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(!viewModel.isSpeaking)
            }
        }
    }

    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Progress")
                .font(.headline)
            HStack {
                Text("Elapsed: \(String(format: "%.1fs", viewModel.elapsed))")
                Spacer()
                levelMeter(level: viewModel.level)
                    .frame(width: 120, height: 8)
            }
        }
    }

    private func levelMeter(level: Float) -> some View {
        GeometryReader { geometry in
            let normalized = max(0, min(1, CGFloat(level * 2)))
            RoundedRectangle(cornerRadius: 4)
                .fill(Color.green.opacity(0.6))
                .frame(width: geometry.size.width * normalized)
                .background(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(Color.gray.opacity(0.2))
                )
        }
    }

    private var metricsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Metrics")
                    .font(.headline)
                Spacer()
                Button("Clear") { viewModel.resetMetrics() }
                    .font(.caption)
            }
            if viewModel.metricsHistory.isEmpty {
                Text("Speak to collect metrics.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.metricsHistory) { metrics in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(metrics.timestamp.formatted(date: .omitted, time: .standard))
                                    .font(.caption.bold())
                                if let cold = metrics.coldLoadDuration {
                                    Text("Cold Load: \(String(format: "%.0f ms", cold * 1000))")
                                        .font(.caption2)
                                }
                                if let warm = metrics.warmLoadDuration {
                                    Text("Warm Load: \(String(format: "%.0f ms", warm * 1000))")
                                        .font(.caption2)
                                }
                                if let total = metrics.totalDuration {
                                    Text("Total: \(String(format: "%.0f ms", total * 1000))")
                                        .font(.caption2)
                                }
                                if let memory = metrics.peakMemoryMB {
                                    Text("Peak RAM: \(String(format: "%.1f MB", memory))")
                                        .font(.caption2)
                                }
                                if !metrics.segmentDurations.isEmpty {
                                    Text("Segments: \(metrics.segmentDurations.map { String(format: "%.0f", $0 * 1000) }.joined(separator: ", ")) ms")
                                        .font(.caption2)
                                }
                            }
                            .padding(8)
                            .background(RoundedRectangle(cornerRadius: 8).fill(Color.gray.opacity(0.1)))
                        }
                    }
                }
            }
        }
    }

    private var keyboardToolbar: some View {
        HStack {
            Spacer()
            Button("Done") { isTextFocused = false }
        }
    }
}

#Preview {
    NavigationStack {
        TTSView()
    }
}
