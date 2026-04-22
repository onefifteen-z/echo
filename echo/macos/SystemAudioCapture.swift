import Foundation
import AVFoundation
@preconcurrency import ScreenCaptureKit
import CoreMedia
import CoreGraphics

func requestScreenRecordingAccess() throws {
    if CGPreflightScreenCaptureAccess() {
        return
    }
    if !CGRequestScreenCaptureAccess() {
        throw CaptureError.permissionDenied
    }
}

func runCapture() async throws {
    let content = try await SCShareableContent.excludingDesktopWindows(
        false,
        onScreenWindowsOnly: true
    )
    guard let display = content.displays.first else {
        throw CaptureError.noDisplay
    }

    let filter = SCContentFilter(
        display: display,
        excludingApplications: [],
        exceptingWindows: []
    )

    let config = SCStreamConfiguration()
    config.capturesAudio = true
    config.excludesCurrentProcessAudio = true
    config.sampleRate = 24_000
    config.channelCount = 2
    config.minimumFrameInterval = CMTime(value: 1, timescale: 30)
    config.width = 2
    config.height = 2
    if #available(macOS 15.0, *) {
        config.captureMicrophone = false
    }

    let dumper = AudioDumper()
    let stream = SCStream(filter: filter, configuration: config, delegate: dumper)
    try stream.addStreamOutput(
        dumper,
        type: .audio,
        sampleHandlerQueue: DispatchQueue(label: "echo.systemaudio.output")
    )
    try await stream.startCapture()

    while true {
        try await Task.sleep(nanoseconds: 1_000_000_000)
    }
}

enum CaptureError: Error {
    case permissionDenied
    case noDisplay
}

final class AudioDumper: NSObject, SCStreamDelegate, SCStreamOutput {
    private var converter: AVAudioConverter?
    private var outputFormat: AVAudioFormat?

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio else { return }

        do {
            try sampleBuffer.withAudioBufferList { audioBufferList, _ in
                guard let fmt = CMSampleBufferGetFormatDescription(sampleBuffer),
                      let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(fmt) else {
                    return
                }
                let asbd = asbdPtr.pointee

                if converter == nil {
                    guard let srcFormat = AVAudioFormat(
                        commonFormat: .pcmFormatFloat32,
                        sampleRate: asbd.mSampleRate,
                        channels: asbd.mChannelsPerFrame,
                        interleaved: false
                    ),
                    let targetFormat = AVAudioFormat(
                        commonFormat: .pcmFormatInt16,
                        sampleRate: 24_000,
                        channels: asbd.mChannelsPerFrame,
                        interleaved: true
                    ) else {
                        return
                    }
                    outputFormat = targetFormat
                    converter = AVAudioConverter(from: srcFormat, to: targetFormat)
                }

                guard let converter = converter, let outFmt = outputFormat else { return }
                guard let srcBuffer = AVAudioPCMBuffer(
                    pcmFormat: converter.inputFormat,
                    frameCapacity: AVAudioFrameCount(sampleBuffer.numSamples)
                ) else { return }
                srcBuffer.frameLength = srcBuffer.frameCapacity
                guard let channelData = srcBuffer.floatChannelData else { return }

                let channels = min(
                    Int(converter.inputFormat.channelCount),
                    Int(audioBufferList.count)
                )
                let frameLen = Int(srcBuffer.frameLength)
                let requiredBytes = frameLen * MemoryLayout<Float>.size

                for i in 0..<channels {
                    let buffer = audioBufferList[i]
                    guard let raw = buffer.mData else { continue }
                    let available = Int(buffer.mDataByteSize)
                    let copyBytes = min(requiredBytes, available)
                    memcpy(channelData[i], raw, copyBytes)
                    if copyBytes < requiredBytes {
                        let remain = requiredBytes - copyBytes
                        memset(channelData[i].advanced(by: copyBytes / MemoryLayout<Float>.size), 0, remain)
                    }
                }

                let outCap = AVAudioFrameCount(
                    ceil(Double(srcBuffer.frameLength) * outFmt.sampleRate / converter.inputFormat.sampleRate)
                )
                guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: outCap) else { return }

                var convErr: NSError?
                let status = converter.convert(to: outBuffer, error: &convErr) { _, outStatus in
                    outStatus.pointee = .haveData
                    return srcBuffer
                }
                guard status != .error,
                      outBuffer.frameLength > 0,
                      let ptr = outBuffer.int16ChannelData?[0] else {
                    return
                }

                let bytes = Int(outBuffer.frameLength) * Int(outFmt.streamDescription.pointee.mBytesPerFrame)
                let data = Data(bytes: ptr, count: bytes)
                FileHandle.standardOutput.write(data)
            }
        } catch {
            fputs("audio processing error: \(error)\n", stderr)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("stream stopped: \(error)\n", stderr)
    }
}

Task {
    do {
        try requestScreenRecordingAccess()
        try await runCapture()
    } catch {
        fputs("echo-system-audio-capture error: \(error)\n", stderr)
        exit(1)
    }
}

dispatchMain()
