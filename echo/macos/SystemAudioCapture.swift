import Foundation
import AVFoundation
import CoreAudio
import AudioToolbox
@preconcurrency import ScreenCaptureKit
import CoreMedia
import CoreGraphics

let kEchoOutputSampleRate: Float64 = 24_000
let kEchoOutputChannels: AVAudioChannelCount = 2

enum CaptureError: Error {
    case permissionDenied
    case noDisplay
    case tapUnavailable(String)
    case tapCreateFailed(OSStatus)
    case aggregateCreateFailed(OSStatus)
    case ioProcSetupFailed(OSStatus)
    case formatSetupFailed
}

func requestScreenRecordingAccess() throws {
    if CGPreflightScreenCaptureAccess() {
        return
    }
    if !CGRequestScreenCaptureAccess() {
        throw CaptureError.permissionDenied
    }
}

// MARK: - Audio-only path (System Audio Recording Only, macOS 14.2+)

@available(macOS 14.2, *)
final class AudioOnlyTapSession {
    var tapID: AudioObjectID = AudioObjectID(kAudioObjectUnknown)
    var aggID: AudioDeviceID = AudioDeviceID(kAudioObjectUnknown)
    var ioProcID: AudioDeviceIOProcID?
    var converter: AVAudioConverter?
    var srcFormat: AVAudioFormat?
    var dstFormat: AVAudioFormat?

    func start() throws {
        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        description.uuid = UUID()
        description.muteBehavior = .unmuted
        description.isPrivate = true
        description.isExclusive = false

        var newTap: AudioObjectID = AudioObjectID(kAudioObjectUnknown)
        let tapStatus = AudioHardwareCreateProcessTap(description, &newTap)
        guard tapStatus == noErr, newTap != AudioObjectID(kAudioObjectUnknown) else {
            throw CaptureError.tapCreateFailed(tapStatus)
        }
        tapID = newTap

        let aggregateUID = UUID().uuidString
        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey as String: "Echo System Audio Tap",
            kAudioAggregateDeviceUIDKey as String: aggregateUID,
            kAudioAggregateDeviceMainSubDeviceKey as String: "",
            kAudioAggregateDeviceIsPrivateKey as String: true,
            kAudioAggregateDeviceIsStackedKey as String: false,
            kAudioAggregateDeviceTapListKey as String: [
                [
                    kAudioSubTapUIDKey as String: description.uuid.uuidString,
                    kAudioSubTapDriftCompensationKey as String: false,
                ]
            ],
        ]

        var newAgg: AudioDeviceID = AudioDeviceID(kAudioObjectUnknown)
        let aggStatus = AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &newAgg)
        guard aggStatus == noErr, newAgg != AudioDeviceID(kAudioObjectUnknown) else {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
            throw CaptureError.aggregateCreateFailed(aggStatus)
        }
        aggID = newAgg

        let virtualFormat = try Self.firstInputVirtualFormat(of: aggID)
        var asbd = virtualFormat
        guard let src = AVAudioFormat(streamDescription: &asbd),
              let dst = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: kEchoOutputSampleRate,
                channels: kEchoOutputChannels,
                interleaved: true
              ),
              let conv = AVAudioConverter(from: src, to: dst) else {
            cleanup()
            throw CaptureError.formatSetupFailed
        }
        srcFormat = src
        dstFormat = dst
        converter = conv

        var procID: AudioDeviceIOProcID?
        let createStatus = AudioDeviceCreateIOProcIDWithBlock(
            &procID,
            aggID,
            nil
        ) { [weak self] _, inInputData, _, _, _ in
            self?.handleInput(inInputData)
        }
        guard createStatus == noErr, let pid = procID else {
            cleanup()
            throw CaptureError.ioProcSetupFailed(createStatus)
        }
        ioProcID = pid

        let startStatus = AudioDeviceStart(aggID, pid)
        guard startStatus == noErr else {
            AudioDeviceDestroyIOProcID(aggID, pid)
            ioProcID = nil
            cleanup()
            throw CaptureError.ioProcSetupFailed(startStatus)
        }

        fputs("echo-system-audio-capture: using audio-only tap path (System Audio Recording Only)\n", stderr)
    }

    private static func firstInputVirtualFormat(of device: AudioObjectID) throws -> AudioStreamBasicDescription {
        var streamsAddr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        var streamSize: UInt32 = 0
        var status = AudioObjectGetPropertyDataSize(device, &streamsAddr, 0, nil, &streamSize)
        guard status == noErr, streamSize > 0 else {
            throw CaptureError.tapUnavailable("no input streams on aggregate (status=\(status))")
        }
        let streamCount = Int(streamSize) / MemoryLayout<AudioObjectID>.size
        var streams = [AudioObjectID](repeating: AudioObjectID(kAudioObjectUnknown), count: streamCount)
        status = AudioObjectGetPropertyData(device, &streamsAddr, 0, nil, &streamSize, &streams)
        guard status == noErr, let firstStream = streams.first, firstStream != AudioObjectID(kAudioObjectUnknown) else {
            throw CaptureError.tapUnavailable("could not list aggregate streams (status=\(status))")
        }

        var virtualFormat = AudioStreamBasicDescription()
        var fmtSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var fmtAddr = AudioObjectPropertyAddress(
            mSelector: kAudioStreamPropertyVirtualFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        status = AudioObjectGetPropertyData(firstStream, &fmtAddr, 0, nil, &fmtSize, &virtualFormat)
        guard status == noErr, virtualFormat.mSampleRate > 0 else {
            throw CaptureError.tapUnavailable("could not read input virtual format (status=\(status))")
        }
        return virtualFormat
    }

    private func handleInput(_ inInputData: UnsafePointer<AudioBufferList>) {
        guard let srcFormat = srcFormat,
              let dstFormat = dstFormat,
              let converter = converter else { return }

        let abl = UnsafeMutableAudioBufferListPointer(
            UnsafeMutablePointer<AudioBufferList>(mutating: inInputData)
        )
        guard let firstBuf = abl.first, firstBuf.mDataByteSize > 0 else {
            return
        }
        let bytesPerFrame = max(1, Int(srcFormat.streamDescription.pointee.mBytesPerFrame))
        let frames = Int(firstBuf.mDataByteSize) / bytesPerFrame
        guard frames > 0 else { return }

        guard let inBuf = AVAudioPCMBuffer(
            pcmFormat: srcFormat,
            frameCapacity: AVAudioFrameCount(frames)
        ) else { return }
        inBuf.frameLength = AVAudioFrameCount(frames)

        if srcFormat.commonFormat == .pcmFormatFloat32 && !srcFormat.isInterleaved {
            let channels = min(Int(srcFormat.channelCount), Int(abl.count))
            for c in 0..<channels {
                let buf = abl[c]
                guard let src = buf.mData,
                      let dst = inBuf.floatChannelData?[c] else { continue }
                let copyBytes = min(Int(buf.mDataByteSize), frames * MemoryLayout<Float>.size)
                memcpy(dst, src, copyBytes)
            }
        } else if srcFormat.commonFormat == .pcmFormatFloat32 && srcFormat.isInterleaved {
            guard let src = abl[0].mData,
                  let dst = inBuf.floatChannelData?[0] else { return }
            let copyBytes = min(
                Int(abl[0].mDataByteSize),
                frames * Int(srcFormat.streamDescription.pointee.mBytesPerFrame)
            )
            memcpy(dst, src, copyBytes)
        } else {
            return
        }

        let outCap = AVAudioFrameCount(
            ceil(Double(frames) * dstFormat.sampleRate / srcFormat.sampleRate) + 64
        )
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: outCap) else { return }
        var convError: NSError?
        let status = converter.convert(to: outBuf, error: &convError) { _, outStatus in
            outStatus.pointee = .haveData
            return inBuf
        }
        guard status != .error,
              outBuf.frameLength > 0,
              let ptr = outBuf.int16ChannelData?[0] else {
            return
        }

        let bytes = Int(outBuf.frameLength) * Int(dstFormat.streamDescription.pointee.mBytesPerFrame)
        let data = Data(bytes: ptr, count: bytes)
        FileHandle.standardOutput.write(data)
    }

    func cleanup() {
        if let pid = ioProcID, aggID != AudioDeviceID(kAudioObjectUnknown) {
            AudioDeviceStop(aggID, pid)
            AudioDeviceDestroyIOProcID(aggID, pid)
            ioProcID = nil
        }
        if aggID != AudioDeviceID(kAudioObjectUnknown) {
            AudioHardwareDestroyAggregateDevice(aggID)
            aggID = AudioDeviceID(kAudioObjectUnknown)
        }
        if tapID != AudioObjectID(kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
    }
}

// MARK: - ScreenCaptureKit fallback (Screen Recording permission)

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

// MARK: - Top-level: prefer audio-only tap, fall back to ScreenCaptureKit.

func envAllowsTap() -> Bool {
    let raw = (ProcessInfo.processInfo.environment["ECHO_DISABLE_AUDIO_TAP"] ?? "").lowercased()
    return !["1", "true", "yes", "on"].contains(raw)
}

var retainedTapSession: AnyObject?

Task {
    if #available(macOS 14.2, *), envAllowsTap() {
        let session = AudioOnlyTapSession()
        do {
            try session.start()
            retainedTapSession = session
            // Capture continues on Core Audio's IO thread; just keep the process alive.
            return
        } catch {
            fputs("echo-system-audio-capture: audio-only tap unavailable (\(error)); "
                  + "falling back to ScreenCaptureKit (Screen Recording permission needed)\n", stderr)
        }
    }
    do {
        try requestScreenRecordingAccess()
        try await runCapture()
    } catch {
        fputs("echo-system-audio-capture error: \(error)\n", stderr)
        exit(1)
    }
}

dispatchMain()
