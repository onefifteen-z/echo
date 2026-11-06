// Echo meeting watcher: reports which processes are capturing audio input (macOS 14+ CoreAudio
// process objects) and the on-screen window titles of watched apps.
//
// Usage: meeting-watch [--interval SECONDS] BUNDLE_PREFIX...
// Prints one JSON line whenever the result changes:
//   {"inputs":[{"pid":123,"bundle":"us.zoom.xos"}],"windows":{"us.zoom.xos":["Zoom Meeting"]}}
// or {"error":"..."} once when process objects are unavailable. Exits when stdin closes.

import AppKit
import CoreAudio
import CoreGraphics
import Foundation

struct InputProcess: Codable, Equatable {
    let pid: Int32
    let bundle: String
}

struct Report: Codable, Equatable {
    let inputs: [InputProcess]
    let windows: [String: [String]]
}

func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
}

func processObjects() -> [AudioObjectID]? {
    var addr = address(kAudioHardwarePropertyProcessObjectList)
    var size: UInt32 = 0
    let system = AudioObjectID(kAudioObjectSystemObject)
    guard AudioObjectGetPropertyDataSize(system, &addr, 0, nil, &size) == noErr else { return nil }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &ids) == noErr else { return nil }
    return ids
}

func uint32Property(_ obj: AudioObjectID, _ selector: AudioObjectPropertySelector) -> UInt32? {
    var addr = address(selector)
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    return AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &value) == noErr ? value : nil
}

func pidProperty(_ obj: AudioObjectID) -> pid_t? {
    var addr = address(kAudioProcessPropertyPID)
    var value: pid_t = 0
    var size = UInt32(MemoryLayout<pid_t>.size)
    return AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &value) == noErr ? value : nil
}

func bundleProperty(_ obj: AudioObjectID) -> String? {
    var addr = address(kAudioProcessPropertyBundleID)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &value) == noErr, let v = value else { return nil }
    return v.takeRetainedValue() as String
}

func inputProcesses() -> [InputProcess]? {
    guard let ids = processObjects() else { return nil }
    var out: [InputProcess] = []
    for id in ids where uint32Property(id, kAudioProcessPropertyIsRunningInput) == 1 {
        guard let pid = pidProperty(id) else { continue }
        let bundle = bundleProperty(id)
            ?? NSRunningApplication(processIdentifier: pid)?.bundleIdentifier
            ?? ""
        out.append(InputProcess(pid: pid, bundle: bundle))
    }
    return out.sorted { $0.pid < $1.pid }
}

func windowTitles(prefixes: [String]) -> [String: [String]] {
    guard !prefixes.isEmpty,
          let info = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID)
              as? [[String: Any]]
    else { return [:] }
    var bundleByPid: [pid_t: String] = [:]
    for app in NSWorkspace.shared.runningApplications {
        if let b = app.bundleIdentifier, prefixes.contains(where: { b.hasPrefix($0) }) {
            bundleByPid[app.processIdentifier] = b
        }
    }
    var out: [String: [String]] = [:]
    for w in info {
        guard let pid = w[kCGWindowOwnerPID as String] as? pid_t,
              let bundle = bundleByPid[pid],
              (w[kCGWindowLayer as String] as? Int) == 0,
              let name = w[kCGWindowName as String] as? String,
              !name.isEmpty
        else { continue }
        if !(out[bundle]?.contains(name) ?? false) { out[bundle, default: []].append(name) }
    }
    return out
}

func emit<T: Encodable>(_ value: T) {
    let enc = JSONEncoder()
    enc.outputFormatting = [.sortedKeys]
    if let data = try? enc.encode(value), let line = String(data: data, encoding: .utf8) {
        print(line)
        fflush(stdout)
    }
}

@main
struct MeetingWatch {
    static func main() {
        var args = Array(CommandLine.arguments.dropFirst())
        var interval = 2.0
        if let i = args.firstIndex(of: "--interval"), i + 1 < args.count {
            interval = Double(args[i + 1]) ?? interval
            args.removeSubrange(i...(i + 1))
        }
        let prefixes = args

        // Exit when the parent closes our stdin (or dies).
        Thread.detachNewThread {
            while readLine() != nil {}
            exit(0)
        }

        var last: Report?
        var reportedError = false
        while true {
            if let inputs = inputProcesses() {
                let watched = inputs.contains { p in prefixes.contains { p.bundle.hasPrefix($0) } }
                let report = Report(inputs: inputs, windows: watched ? windowTitles(prefixes: prefixes) : [:])
                if report != last {
                    emit(report)
                    last = report
                }
            } else if !reportedError {
                emit(["error": "CoreAudio process objects are unavailable (macOS 14 or later is required)."])
                reportedError = true
            }
            Thread.sleep(forTimeInterval: interval)
        }
    }
}
