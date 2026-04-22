"""Real-time capture backends for Echo live ASR."""

from __future__ import annotations

import os
import queue
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


def _require_sounddevice() -> None:
    try:
        import sounddevice  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Live capture needs sounddevice. Install ASR extras: pip install -e '.[asr]'"
        ) from e


def _system_audio_helper_source_path() -> Path:
    return Path(__file__).resolve().parent / "macos" / "SystemAudioCapture.swift"


def _system_audio_helper_binary_path() -> Path:
    cache_home = Path(
        os.environ.get("XDG_CACHE_HOME")
        or (Path.home() / ".cache")
    )
    return cache_home / "echo" / "system_audio_capture"


def ensure_macos_system_audio_helper() -> str:
    """Build bundled ScreenCaptureKit helper and return executable path."""
    if sys.platform != "darwin":
        raise RuntimeError("System-authorized live capture is available on macOS only.")

    src = _system_audio_helper_source_path()
    if not src.is_file():
        raise RuntimeError(f"Bundled helper source not found: {src}")

    out = _system_audio_helper_binary_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    needs_build = (not out.exists()) or (src.stat().st_mtime > out.stat().st_mtime)
    if not needs_build:
        return str(out)

    cmd_xcrun = [
        "xcrun",
        "swiftc",
        "-O",
        "-framework",
        "ScreenCaptureKit",
        "-framework",
        "AVFoundation",
        "-framework",
        "CoreMedia",
        "-framework",
        "CoreGraphics",
        str(src),
        "-o",
        str(out),
    ]
    cmd_swiftc = cmd_xcrun[1:]

    proc = subprocess.run(cmd_xcrun, capture_output=True, text=True)
    if proc.returncode != 0:
        proc2 = subprocess.run(cmd_swiftc, capture_output=True, text=True)
        if proc2.returncode != 0:
            err = (proc.stderr or "") + "\n" + (proc2.stderr or "")
            raise RuntimeError(
                "Failed to build macOS system-audio helper with swiftc.\n"
                f"Source: {src}\n"
                f"Errors:\n{err.strip()}"
            )
    out.chmod(0o755)
    return str(out)


def list_input_devices() -> str:
    """Return a human-readable list of host API input devices (for --list-audio-devices)."""
    _require_sounddevice()
    import sounddevice as sd

    lines: list[str] = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue
        lines.append(f"  [{i}] {d.get('name', '?')} — {d.get('default_samplerate', '?')} Hz in")
    return "Input devices:\n" + ("\n".join(lines) if lines else "  (none)")


def resolve_input_device(raw: str | None, *, default: int | None) -> int | None:
    """Parse device index or case-insensitive substring match against input device names."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    s = raw.strip()
    try:
        return int(s)
    except ValueError:
        import sounddevice as sd

        s_low = s.lower()
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) <= 0:
                continue
            name = str(d.get("name", "")).lower()
            if s_low in name:
                return i
        raise ValueError(f"No input device matching substring: {raw!r}")


@dataclass
class LiveMixerConfig:
    mic_device: int | None
    system_device: int | None
    samplerate: int = 16000
    blocksize: int = 2048


@dataclass
class LiveSystemAuthorizedConfig:
    """Capture mic with sounddevice + system output from an external authorized command.

    The command is expected to emit raw PCM to stdout as:
      - sample rate: 24000 Hz
      - format: s16le
      - channels: 2 (stereo)
    This matches SystemAudioDump defaults.
    """

    mic_device: int | None
    command: str
    samplerate: int = 16000
    blocksize: int = 2048


class LivePairedCapture:
    """PortAudio input stream(s): mic only, or mic + second device mixed 50/50 to mono ``out_q``."""

    def __init__(self, cfg: LiveMixerConfig) -> None:
        self.cfg = cfg
        self._stop = threading.Event()
        self.out_q: queue.Queue = queue.Queue(maxsize=128)
        self._mic_q: queue.Queue = queue.Queue(maxsize=64)
        self._sys_q: queue.Queue = queue.Queue(maxsize=64)
        self._mix_thread: threading.Thread | None = None
        self._streams: list[object] = []

    def _safe_put(self, q: queue.Queue, item: object) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def start(self) -> None:
        import numpy as np

        _require_sounddevice()
        import sounddevice as sd

        def mic_cb(indata, frames: int, _time, status) -> None:  # noqa: ARG001
            if self._stop.is_set():
                return
            chunk = np.clip(indata.copy().reshape(-1), -1.0, 1.0)
            if self.cfg.system_device is None:
                self._safe_put(self.out_q, chunk)
            else:
                self._safe_put(self._mic_q, chunk)

        mic_stream = sd.InputStream(
            device=self.cfg.mic_device,
            channels=1,
            dtype="float32",
            samplerate=self.cfg.samplerate,
            blocksize=self.cfg.blocksize,
            callback=mic_cb,
        )
        mic_stream.start()
        self._streams.append(mic_stream)

        if self.cfg.system_device is not None:

            def sys_cb(indata, frames: int, _time, status) -> None:  # noqa: ARG001
                if self._stop.is_set():
                    return
                chunk = np.clip(indata.copy().reshape(-1), -1.0, 1.0)
                self._safe_put(self._sys_q, chunk)

            sys_stream = sd.InputStream(
                device=self.cfg.system_device,
                channels=1,
                dtype="float32",
                samplerate=self.cfg.samplerate,
                blocksize=self.cfg.blocksize,
                callback=sys_cb,
            )
            sys_stream.start()
            self._streams.append(sys_stream)

            def mix_loop() -> None:
                while not self._stop.is_set():
                    try:
                        m = self._mic_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        s = self._sys_q.get(timeout=0.2)
                    except queue.Empty:
                        self._safe_put(self.out_q, m)
                        continue
                    n = min(m.shape[0], s.shape[0])
                    if n <= 0:
                        continue
                    mix = 0.5 * (m[:n] + s[:n])
                    self._safe_put(self.out_q, np.clip(mix, -1.0, 1.0))

            self._mix_thread = threading.Thread(target=mix_loop, name="echo-live-mix", daemon=True)
            self._mix_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for st in self._streams:
            try:
                st.stop()
                st.close()
            except Exception:
                pass
        self._streams.clear()
        if self._mix_thread is not None:
            self._mix_thread.join(timeout=2.0)
            self._mix_thread = None

    def read_block(self, timeout: float = 0.5) -> object | None:
        try:
            return self.out_q.get(timeout=timeout)
        except queue.Empty:
            return None


def _resample_24k_to_16k(samples: object) -> object:
    """Fast linear resample (24k -> 16k) for float32 mono arrays."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = x.shape[0]
    if n < 2:
        return np.zeros((0,), dtype=np.float32)
    m = int(n * 2 / 3)
    if m <= 0:
        return np.zeros((0,), dtype=np.float32)
    pos = np.arange(m, dtype=np.float32) * 1.5
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = pos - i0
    out = x[i0] * (1.0 - frac) + x[i1] * frac
    return out.astype(np.float32, copy=False)


class LiveSystemAuthorizedCapture:
    """Mic + system-output capture via external authorized command (no virtual loopback required)."""

    def __init__(self, cfg: LiveSystemAuthorizedConfig) -> None:
        self.cfg = cfg
        self._stop = threading.Event()
        self.out_q: queue.Queue = queue.Queue(maxsize=128)
        self._mic_q: queue.Queue = queue.Queue(maxsize=64)
        self._sys_q: queue.Queue = queue.Queue(maxsize=64)
        self._mix_thread: threading.Thread | None = None
        self._sys_read_thread: threading.Thread | None = None
        self._sys_err_thread: threading.Thread | None = None
        self._streams: list[object] = []
        self._proc: subprocess.Popen[bytes] | None = None
        self.last_stderr_line: str = ""

    def _safe_put(self, q: queue.Queue, item: object) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def start(self) -> None:
        import numpy as np

        _require_sounddevice()
        import sounddevice as sd

        cmd = shlex.split(self.cfg.command.strip())
        if not cmd:
            raise RuntimeError("Empty system-audio command.")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"System-audio command not found: {cmd[0]!r}. "
                "Install SystemAudioDump (or set --live-system-authorized-cmd)."
            ) from e

        if self._proc.stdout is None or self._proc.stderr is None:
            raise RuntimeError("Failed to start system-audio capture process.")

        def mic_cb(indata, frames: int, _time, status) -> None:  # noqa: ARG001
            if self._stop.is_set():
                return
            chunk = np.clip(indata.copy().reshape(-1), -1.0, 1.0)
            self._safe_put(self._mic_q, chunk)

        mic_stream = sd.InputStream(
            device=self.cfg.mic_device,
            channels=1,
            dtype="float32",
            samplerate=self.cfg.samplerate,
            blocksize=self.cfg.blocksize,
            callback=mic_cb,
        )
        mic_stream.start()
        self._streams.append(mic_stream)

        def sys_read_loop() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            raw_buf = bytearray()
            out_buf = np.zeros((0,), dtype=np.float32)
            bytes_per_frame = 4  # 2ch * s16
            while not self._stop.is_set():
                data = self._proc.stdout.read(4096)
                if not data:
                    break
                raw_buf.extend(data)
                n_frames = len(raw_buf) // bytes_per_frame
                if n_frames <= 0:
                    continue
                n_bytes = n_frames * bytes_per_frame
                frame_bytes = bytes(raw_buf[:n_bytes])
                del raw_buf[:n_bytes]
                pcm = np.frombuffer(frame_bytes, dtype="<i2").reshape(-1, 2).astype(np.float32)
                mono_24k = (pcm[:, 0] + pcm[:, 1]) * (0.5 / 32768.0)
                mono_16k = _resample_24k_to_16k(mono_24k)
                if mono_16k.size == 0:
                    continue
                out_buf = np.concatenate([out_buf, mono_16k.astype(np.float32, copy=False)])
                while out_buf.size >= self.cfg.blocksize:
                    blk = out_buf[: self.cfg.blocksize]
                    out_buf = out_buf[self.cfg.blocksize :]
                    self._safe_put(self._sys_q, np.clip(blk, -1.0, 1.0))

        def sys_err_loop() -> None:
            assert self._proc is not None and self._proc.stderr is not None
            while not self._stop.is_set():
                ln = self._proc.stderr.readline()
                if not ln:
                    break
                self.last_stderr_line = ln.decode("utf-8", errors="replace").strip()

        def mix_loop() -> None:
            while not self._stop.is_set():
                try:
                    m = self._mic_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    s = self._sys_q.get(timeout=0.2)
                except queue.Empty:
                    self._safe_put(self.out_q, m)
                    continue
                n = min(m.shape[0], s.shape[0])
                if n <= 0:
                    continue
                mix = 0.5 * (m[:n] + s[:n])
                self._safe_put(self.out_q, np.clip(mix, -1.0, 1.0))

        self._sys_read_thread = threading.Thread(target=sys_read_loop, name="echo-live-sys-read", daemon=True)
        self._sys_err_thread = threading.Thread(target=sys_err_loop, name="echo-live-sys-err", daemon=True)
        self._mix_thread = threading.Thread(target=mix_loop, name="echo-live-sys-mix", daemon=True)
        self._sys_read_thread.start()
        self._sys_err_thread.start()
        self._mix_thread.start()

    def process_exited(self) -> bool:
        return self._proc is not None and self._proc.poll() is not None

    def process_error_text(self) -> str:
        rc = None if self._proc is None else self._proc.poll()
        tail = self.last_stderr_line.strip()
        if rc is None:
            return ""
        if tail:
            return f"system-audio command exited ({rc}): {tail}"
        return f"system-audio command exited ({rc})"

    def stop(self) -> None:
        self._stop.set()
        for st in self._streams:
            try:
                st.stop()
                st.close()
            except Exception:
                pass
        self._streams.clear()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        for t in (self._mix_thread, self._sys_read_thread, self._sys_err_thread):
            if t is not None:
                t.join(timeout=1.0)
        self._mix_thread = None
        self._sys_read_thread = None
        self._sys_err_thread = None

    def read_block(self, timeout: float = 0.5) -> object | None:
        try:
            return self.out_q.get(timeout=timeout)
        except queue.Empty:
            return None
