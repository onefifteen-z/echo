import AVFoundation
import CoreMedia
import Foundation
import Speech

// Echo on-device speech helper.
//
//   --status --locales ja_JP,en_US      one JSON object: {"ja_JP": {"engine":..., "status":...}}
//   --install --locale ja_JP            JSON lines: progress..., then installed | error
//   --stream --locale ja_JP             stdin: raw s16le mono 16 kHz PCM (until EOF)
//                                       stdout: JSON lines ready | partial | final | error
//
// macOS 26+ uses SpeechAnalyzer + SpeechTranscriber (model assets managed by AssetInventory).
// Older systems fall back to SFSpeechRecognizer with on-device buffer recognition.

let kInputSampleRate: Double = 16_000
let stdoutLock = NSLock()

func emit(_ obj: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []) else { return }
    stdoutLock.lock()
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
    stdoutLock.unlock()
}

func fail(_ message: String) -> Never {
    emit(["type": "error", "message": message])
    exit(1)
}

func cmSeconds(_ t: CMTime) -> Double {
    t.isValid && t.isNumeric ? CMTimeGetSeconds(t) : 0
}

/// Reads raw s16le PCM from stdin on a background thread and hands out 16 kHz Int16 buffers.
final class StdinPCMReader: @unchecked Sendable {
    let format = AVAudioFormat(
        commonFormat: .pcmFormatInt16, sampleRate: kInputSampleRate, channels: 1, interleaved: true
    )!
    private var carry = Data()

    func run(onBuffer: @escaping (AVAudioPCMBuffer, Int64) -> Void, onEOF: @escaping () -> Void) {
        let thread = Thread { [self] in
            var samplesSoFar: Int64 = 0
            let handle = FileHandle.standardInput
            while true {
                let chunk = handle.availableData
                if chunk.isEmpty { break }
                carry.append(chunk)
                let usable = carry.count - (carry.count % 2)
                if usable < 3200 { continue }  // ~100 ms
                let frames = AVAudioFrameCount(usable / 2)
                guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { continue }
                buf.frameLength = frames
                carry.prefix(usable).withUnsafeBytes { raw in
                    if let dst = buf.int16ChannelData?[0] {
                        raw.copyBytes(to: UnsafeMutableRawBufferPointer(start: dst, count: usable))
                    }
                }
                carry.removeFirst(usable)
                onBuffer(buf, samplesSoFar)
                samplesSoFar += Int64(frames)
            }
            readerFinished.set()
            onEOF()
        }
        thread.start()
    }
}

func convert(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat, using converter: AVAudioConverter?) -> AVAudioPCMBuffer? {
    guard let converter else { return buffer }
    let ratio = format.sampleRate / buffer.format.sampleRate
    let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 32)
    guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return nil }
    var consumed = false
    var error: NSError?
    converter.convert(to: out, error: &error) { _, status in
        if consumed {
            status.pointee = .noDataNow
            return nil
        }
        consumed = true
        status.pointee = .haveData
        return buffer
    }
    return error == nil ? out : nil
}

// MARK: - SpeechAnalyzer (macOS 26+)

@available(macOS 26.0, *)
func analyzerTranscriber(for localeId: String) async -> SpeechTranscriber? {
    guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: localeId)) else {
        return nil
    }
    return SpeechTranscriber(
        locale: locale,
        transcriptionOptions: [],
        reportingOptions: [.volatileResults, .fastResults],
        attributeOptions: [.audioTimeRange]
    )
}

@available(macOS 26.0, *)
func analyzerStatus(_ localeId: String) async -> String {
    guard SpeechTranscriber.isAvailable, let t = await analyzerTranscriber(for: localeId) else {
        return "unsupported"
    }
    switch await AssetInventory.status(forModules: [t]) {
    case .installed: return "installed"
    case .downloading: return "downloading"
    case .supported: return "supported"
    default: return "unsupported"
    }
}

@available(macOS 26.0, *)
func analyzerInstall(_ localeId: String) async {
    guard let t = await analyzerTranscriber(for: localeId) else { fail("Locale \(localeId) is not supported") }
    do {
        _ = try? await AssetInventory.reserve(locale: t.selectedLocales.first ?? Locale(identifier: localeId))
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [t]) {
            let progress = request.progress
            let observer = progress.observe(\.fractionCompleted) { p, _ in
                emit(["type": "progress", "fraction": p.fractionCompleted])
            }
            try await request.downloadAndInstall()
            observer.invalidate()
        }
        emit(["type": "installed", "locale": localeId])
    } catch {
        fail("Install failed: \(error.localizedDescription)")
    }
}

@available(macOS 26.0, *)
func analyzerStream(_ localeId: String) async {
    guard let transcriber = await analyzerTranscriber(for: localeId) else {
        fail("Locale \(localeId) is not supported by SpeechTranscriber")
    }
    if await AssetInventory.status(forModules: [transcriber]) != .installed {
        fail("Speech model for \(localeId) is not installed. Run the helper with --install --locale \(localeId).")
    }
    let reader = StdinPCMReader()
    let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
        compatibleWith: [transcriber], considering: reader.format
    ) ?? reader.format
    let converter = analyzerFormat == reader.format ? nil : AVAudioConverter(from: reader.format, to: analyzerFormat)

    let (inputs, continuation) = AsyncStream<AnalyzerInput>.makeStream()
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    do {
        try await analyzer.prepareToAnalyze(in: analyzerFormat)
    } catch {
        fail("prepareToAnalyze failed: \(error.localizedDescription)")
    }

    let resultsTask = Task {
        do {
            for try await result in transcriber.results {
                let text = String(result.text.characters)
                emit([
                    "type": result.isFinal ? "final" : "partial",
                    "text": text,
                    "start": cmSeconds(result.range.start),
                    "end": cmSeconds(result.range.end),
                ])
            }
        } catch {
            emit(["type": "error", "message": "results: \(error.localizedDescription)"])
        }
    }

    reader.run(
        onBuffer: { buf, startSample in
            guard let converted = convert(buf, to: analyzerFormat, using: converter) else { return }
            let start = CMTime(value: startSample, timescale: CMTimeScale(kInputSampleRate))
            continuation.yield(AnalyzerInput(buffer: converted, bufferStartTime: start))
        },
        onEOF: { continuation.finish() }
    )

    emit(["type": "ready", "engine": "analyzer", "locale": localeId])
    do {
        if let last = try await analyzer.analyzeSequence(inputs) {
            try await analyzer.finalizeAndFinish(through: last)
        } else {
            await analyzer.cancelAndFinishNow()
        }
    } catch {
        fail("analysis failed: \(error.localizedDescription)")
    }
    _ = await resultsTask.result
}

final class AtomicFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var value: Bool {
        lock.lock(); defer { lock.unlock() }
        return flag
    }
    func set() {
        lock.lock(); flag = true; lock.unlock()
    }
}

let readerFinished = AtomicFlag()

// MARK: - SFSpeechRecognizer fallback (macOS < 26)

func requestLegacyAuthorization() async -> Bool {
    await withCheckedContinuation { cont in
        SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0 == .authorized) }
    }
}

func legacyStatus(_ localeId: String) -> String {
    guard let r = SFSpeechRecognizer(locale: Locale(identifier: localeId)) else { return "unsupported" }
    return r.supportsOnDeviceRecognition ? "installed" : "supported"
}

final class LegacyStreamer: @unchecked Sendable {
    let recognizer: SFSpeechRecognizer
    var request: SFSpeechAudioBufferRecognitionRequest?
    var task: SFSpeechRecognitionTask?
    var segmentOffset: Double = 0
    var lastSampleTime: Double = 0
    let lock = NSLock()

    init(recognizer: SFSpeechRecognizer) {
        self.recognizer = recognizer
    }

    func startTask() {
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.requiresOnDeviceRecognition = true
        req.shouldReportPartialResults = true
        request = req
        segmentOffset = lastSampleTime
        let offset = segmentOffset
        task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            guard let self else { return }
            if let result {
                let segs = result.bestTranscription.segments
                let start = offset + (segs.first?.timestamp ?? 0)
                let end = offset + ((segs.last.map { $0.timestamp + $0.duration }) ?? 0)
                emit([
                    "type": result.isFinal ? "final" : "partial",
                    "text": result.bestTranscription.formattedString,
                    "start": start,
                    "end": end,
                ])
                if result.isFinal { self.restart() }
            } else if error != nil {
                self.restart()
            }
        }
    }

    func restart() {
        lock.lock(); defer { lock.unlock() }
        task = nil
        if readerFinished.value { return }
        startTask()
    }

    func append(_ buf: AVAudioPCMBuffer, startSample: Int64) {
        lock.lock(); defer { lock.unlock() }
        lastSampleTime = Double(startSample) / kInputSampleRate
        request?.append(buf)
    }

    func finish() {
        lock.lock(); defer { lock.unlock() }
        request?.endAudio()
    }
}

func legacyStream(_ localeId: String) async {
    guard await requestLegacyAuthorization() else {
        fail("Speech Recognition permission denied (System Settings > Privacy & Security > Speech Recognition).")
    }
    guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeId)),
          recognizer.supportsOnDeviceRecognition
    else {
        fail("On-device recognition is unavailable for \(localeId).")
    }
    let streamer = LegacyStreamer(recognizer: recognizer)
    streamer.startTask()
    let reader = StdinPCMReader()
    let floatFormat = AVAudioFormat(standardFormatWithSampleRate: kInputSampleRate, channels: 1)!
    let converter = AVAudioConverter(from: reader.format, to: floatFormat)
    reader.run(
        onBuffer: { buf, start in
            if let f = convert(buf, to: floatFormat, using: converter) { streamer.append(f, startSample: start) }
        },
        onEOF: { streamer.finish() }
    )
    emit(["type": "ready", "engine": "legacy", "locale": localeId])
    while !readerFinished.value {
        try? await Task.sleep(nanoseconds: 200_000_000)
    }
    try? await Task.sleep(nanoseconds: 1_500_000_000)
}

// MARK: - CLI

@main
struct SpeechTranscribeCLI {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        func value(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        let locale = (value("--locale") ?? "ja_JP").replacingOccurrences(of: "-", with: "_")

        if args.contains("--status") {
            let locales = (value("--locales") ?? locale).split(separator: ",").map {
                String($0).replacingOccurrences(of: "-", with: "_")
            }
            var out: [String: Any] = [:]
            for loc in locales {
                if #available(macOS 26.0, *) {
                    out[loc] = ["engine": "analyzer", "status": await analyzerStatus(loc)]
                } else {
                    out[loc] = ["engine": "legacy", "status": legacyStatus(loc)]
                }
            }
            emit(out)
            exit(0)
        }
        if args.contains("--install") {
            if #available(macOS 26.0, *) {
                await analyzerInstall(locale)
                exit(0)
            }
            fail("Programmatic install needs macOS 26+. Add the language under System Settings > Keyboard > Dictation.")
        }
        if args.contains("--stream") {
            let engine = value("--engine") ?? "auto"
            if #available(macOS 26.0, *), engine != "legacy" {
                await analyzerStream(locale)
            } else {
                await legacyStream(locale)
            }
            exit(0)
        }
        fputs("Usage: speech_transcribe --status [--locales a,b] | --install --locale L | --stream --locale L\n", stderr)
        exit(2)
    }
}
