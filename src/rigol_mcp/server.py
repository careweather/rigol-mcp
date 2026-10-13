"""Rigol DS1000Z MCP server."""

import base64
import json
import os
from datetime import datetime
from pathlib import Path


import asyncio
import time

from dotenv import load_dotenv

# Load configuration from a local .env file (e.g. RIGOL_IP, RIGOL_USB, RIGOL_USB_SERIAL)
# before anything reads os.environ. Existing environment variables (e.g. those passed via
# the MCP client's `env` block) take precedence and are not overridden.
load_dotenv()

import mcp.types as types
import pyvisa
from mcp.server import Server
from mcp.server.stdio import stdio_server

from rigol_mcp.waveform_analysis import describe_waveform as _describe_waveform
from rigol_mcp.memory_export import get_acquisition_info, export_acquisition
from rigol_mcp.scope import (
    get_scope, invalidate_scope, disconnect_scope, usb_in_use, active_backend,
    screenshot_png,
    get_cursor_mode, set_cursor_mode, set_cursor_positions, get_cursor_values,
    send_raw, check_scpi_error,
    run, stop, single, autoscale,
    idn, connection_info, set_driver_from_idn,
    measure, measure_between, MEASURE_ITEMS, MEASURE_ITEMS_TWO_SOURCE,
    get_scope_state, set_channel, set_timebase, set_trigger, get_waveform, get_samples,
)
from rigol_mcp import spi_decode

_CHANNELS = ["CHAN1", "CHAN2", "CHAN3", "CHAN4"]

server = Server(
    "rigol-mcp",
    instructions=(
        "Rigol oscilloscope control over SCPI.\n"
        "- Read the scope with the data tools first: measure/measure_between for numeric "
        "readings, get_waveform for trace shape/frequency/amplitude analysis, "
        "get_scope_state for configuration. They return compact structured text that is "
        "far cheaper and easier to reason over than an image.\n"
        "- screenshot is a fallback for genuinely visual checks only (on-screen menus, "
        "cursor placement, confirming what a human sees) — do not use it as the default "
        "way to inspect signals.\n"
        "- Call tools strictly sequentially, never concurrently: all commands share one "
        "instrument connection."
    ),
)

# Serialises all VISA operations — the underlying TCP socket is not thread/async safe.
_scope_lock = asyncio.Lock()
_last_call_time: float = 0.0
_MIN_INTERVAL = 0.1          # 100 ms minimum between SCPI operations
_POST_SCREENSHOT_DELAY = 2.0 # scope needs recovery time after large display transfer
_MAX_ATTEMPTS = 3            # initial try + 2 reconnect-and-retry attempts
_RETRY_BACKOFF = 0.2         # seconds to let a flaky USB link settle before retrying

# Communication faults that warrant a reconnect-and-retry. Other exceptions (e.g. bad
# arguments, SCPI errors) are bugs/usage errors and must propagate unretried.
_RETRYABLE = (pyvisa.errors.VisaIOError, UnicodeDecodeError, OSError)


def _is_retryable(exc: BaseException) -> bool:
    """Whether exc is a communication fault rather than a usage error.

    pyvisa-py's USB write re-raises usb.core.USBError as ValueError(str(e)), whose text
    starts with "[Errno N]". A scope that re-enumerated (replug, power cycle) leaves the
    cached session failing this way with ENODEV, so those ValueErrors reconnect too.
    """
    if isinstance(exc, _RETRYABLE):
        return True
    return isinstance(exc, ValueError) and str(exc).startswith("[Errno ")

# send_raw sends arbitrary SCPI and can put the scope in any state, so it is opt-in:
# it is only advertised and accepted when RIGOL_ENABLE_SEND_RAW is set to a truthy value.
_SEND_RAW_ENV = "RIGOL_ENABLE_SEND_RAW"


def _send_raw_enabled() -> bool:
    return os.environ.get(_SEND_RAW_ENV, "").strip().lower() not in ("", "0", "false", "no", "off")


async def _call(fn, *args, _retry=True, **kwargs):
    """Call fn(scope, *args, **kwargs) with the cached connection.

    Serialises concurrent calls via a lock, enforces a minimum inter-command gap, and
    recovers from communication faults by reconnecting and retrying (up to _MAX_ATTEMPTS).
    USBTMC links in particular occasionally drop the first access after opening or after a
    large transfer; a reconnect on a fresh session clears it. The last failure propagates.
    """
    global _last_call_time
    async with _scope_lock:
        elapsed = time.monotonic() - _last_call_time
        if elapsed < _MIN_INTERVAL:
            await asyncio.sleep(_MIN_INTERVAL - elapsed)
        try:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                try:
                    return fn(get_scope(), *args, **kwargs)
                except Exception as exc:
                    if not _is_retryable(exc):
                        raise
                    invalidate_scope()  # drop the session so the next attempt reconnects
                    if not _retry or attempt == _MAX_ATTEMPTS:
                        raise
                    await asyncio.sleep(_RETRY_BACKOFF)
        finally:
            _last_call_time = time.monotonic()


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    tools = [
        types.Tool(
            name="disconnect",
            description="Release this server's cached scope connection without opening one or "
                        "sending acquisition commands. Waits for any current scope operation to "
                        "finish. A later scope tool may reconnect. Does not disconnect other servers.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="get_acquisition_info",
            description="Read acquisition status, sample rate, memory-depth setting, enabled channels, "
                        "and implemented transfer modes. Does not arm or stop the scope. AUTO memory "
                        "depth is a setting; export preambles report actual record lengths.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="export_acquisition",
            description="Export the complete stopped acquisition memory to uint8 binary files and a "
                        "JSON manifest (DS1000Z). Returns paths and compact integrity/timing summaries. "
                        "Defaults to all enabled analog channels. Requires STOP; never stops, rearms, "
                        "enables channels, or falls back to screen data. Keep scope controls untouched "
                        "during export. Partial transfers are saved and explicitly marked incomplete.",
            inputSchema={
                "type": "object", "additionalProperties": False,
                "properties": {
                    "channels": {"type": "array", "minItems": 1, "maxItems": 4,
                                 "uniqueItems": True, "items": {"type": "string",
                                 "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"]}},
                    "output_dir": {"type": "string", "description": "Local parent directory; "
                                   "defaults to RIGOL_WAVEFORM_DIR or waveforms/. A unique export "
                                   "subdirectory is created; existing captures are never overwritten."},
                },
            },
        ),
        types.Tool(
            name="screenshot",
            description=(
                "Capture a screenshot of the oscilloscope display. "
                "Returns the image and the absolute path where the PNG was saved. "
                "This is a fallback, not the primary way to read the scope: for numeric "
                "readings use measure/measure_between and for trace data use get_waveform — "
                "they return compact structured text that is faster and cheaper to reason "
                "over than an image. Reach for screenshot only when a genuinely visual check "
                "is needed (on-screen menus/UI state, cursor placement, display rendering, "
                "or confirming what a human sees). "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="idn",
            description=(
                "Identify the instrument and report connection details. "
                "Always returns the connection block (transport, RIGOL_USB/RIGOL_IP env "
                "vars, backend hint, resource string, session state, and the detected "
                "dialect driver — DS1000Z, DHO, …) followed by the scope's *IDN? string. "
                "If the *IDN? query fails, the connection block is still returned with "
                "the error — use it to spot LAN-vs-USB misconfig or an unreachable IP "
                "before assuming the scope itself is the problem. "
                "Call this first to verify connectivity and confirm the correct driver "
                "was selected. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="get_scope_state",
            description=(
                "Return a snapshot of the scope's current configuration: "
                "active channels (scale, offset, coupling, probe), timebase, and trigger. "
                "Call this at the start of a session to understand the current setup. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="set_channel",
            description=(
                "Configure a channel. Only specified parameters are changed. "
                "channel: CHAN1–CHAN4. "
                "scale_v_div: V/div. offset_v: volts. coupling: AC, DC, or GND. "
                "probe: attenuation ratio (1, 10, 100, …). "
                "Parameter names match get_scope_state output for easy round-tripping. "
                "Returns the resulting channel configuration. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "channel":    {"type": "string", "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"]},
                    "display":    {"type": "boolean", "description": "Turn channel on/off"},
                    "scale_v_div": {"type": ["number", "string"], "description": "Vertical scale in V/div"},
                    "offset_v":   {"type": ["number", "string"], "description": "Vertical offset in volts"},
                    "coupling":   {"type": "string", "enum": ["AC", "DC", "GND"]},
                    "probe":      {"type": ["number", "string"], "description": "Probe attenuation ratio (e.g. 1, 10, 100)"},
                },
                "required": ["channel"],
            },
        ),
        types.Tool(
            name="set_timebase",
            description=(
                "Set the horizontal timebase. "
                "scale_s_div: seconds per division (e.g. 0.001 for 1 ms/div). "
                "offset_s: shifts the display window; time_start = offset_s − 6×scale_s_div, time_end = offset_s + 6×scale_s_div. "
                "Trigger (t=0) is always a zero crossing when using edge trigger. "
                "To align the right edge to a zero crossing at time T: set offset_s = T − 6×scale_s_div. "
                "To put the trigger at the left edge of the screen: set offset_s = +6×scale_s_div. "
                "Parameter names match get_scope_state output for easy round-tripping. "
                "Returns the resulting timebase configuration. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scale_s_div": {"type": ["number", "string"], "description": "Time per division in seconds"},
                    "offset_s":    {"type": ["number", "string"], "description": "Trigger offset in seconds"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="set_trigger",
            description=(
                "Configure edge trigger. "
                "source: CHAN1–CHAN4 or EXT. "
                "slope: POS (rising), NEG (falling), or RFAL (either). "
                "level: trigger level in volts. "
                "Returns the resulting trigger configuration. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Trigger source, e.g. CHAN1"},
                    "slope":  {"type": "string", "enum": ["POS", "NEG", "RFAL"]},
                    "level":  {"type": ["number", "string"], "description": "Trigger level in volts"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="measure",
            description=(
                "Query a single-source built-in measurement on a channel. "
                "Preferred over screenshot for reading values — numeric results are "
                "cheaper and easier to analyse than an image. "
                "For stable readings: on DS1000Z, stop acquisition first. "
                "On DHO, keep acquisition running — the DHO measurement engine only populates "
                "item values from live acquisitions; some items (VMAX/VMIN/VTOP/FREQUENCY/…) "
                "return 9.9E37 if first queried on a stopped scope. "
                "channel: CHAN1–CHAN4. "
                "item: VMAX, VMIN, VPP, "
                "VTOP (pulse top flat level, histogram-derived — not the same as VMAX), "
                "VBASE (pulse base flat level — not the same as VMIN), "
                "VAMP (=VTOP−VBASE — not the same as VPP=VMAX−VMIN), "
                "VAVG, VRMS (RMS over screen window), PVRMS (RMS over one period), "
                "VUPPER/VMID/VLOWER (timing thresholds at 90%/50%/10% of VAMP by default), "
                "VARIANCE (statistical variance of voltage samples), "
                "FREQUENCY, PERIOD, PWIDTH, NWIDTH, PDUTY, NDUTY, "
                "RTIME, FTIME, OVERSHOOT, PRESHOOT, "
                "PSLEWRATE, NSLEWRATE (slew rate, V/s), "
                "TVMAX, TVMIN (time position at which VMAX/VMIN occurs), "
                "MAREA (waveform area, V·s over screen window), MPAREA (area per period, V·s), "
                "PPULSES, NPULSES, PEDGES, NEDGES. "
                "A return value of 9.9E37 is the scope's invalid/overflow sentinel — "
                "it means the measurement could not be computed (e.g. FREQUENCY returns 9.9E37 "
                "when the timebase is too narrow to show a complete cycle; widen scale and retry); "
                "such values come back annotated as invalid/overflow. "
                "If the channel's display is OFF it is auto-enabled first (noted in the result). "
                "For delay or phase between two channels use measure_between. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"]},
                    "item":    {"type": "string", "description": "Measurement item (e.g. FREQUENCY, VPP, VRMS)"},
                },
                "required": ["channel", "item"],
            },
        ),
        types.Tool(
            name="measure_between",
            description=(
                "Query a two-source delay or phase measurement between two channels. "
                "source1 is the reference channel, source2 is the measured channel. "
                "DS1000Z items: RDELAY (rising-edge delay, seconds), FDELAY (falling-edge delay, seconds), "
                "RPHASE (rising-edge phase, degrees), FPHASE (falling-edge phase, degrees). "
                "DHO series exposes a 4-way matrix: RRDELAY/RFDELAY/FRDELAY/FFDELAY and "
                "RRPHASE/RFPHASE/FRPHASE/FFPHASE (first letter = source1 edge, second = source2 edge). "
                "On DHO the DS1000Z names are auto-mapped to their homogeneous equivalents "
                "(RDELAY→RRDELAY, FDELAY→FFDELAY, RPHASE→RRPHASE, FPHASE→FFPHASE). "
                "For stable readings: on DS1000Z, stop acquisition first. "
                "On DHO, keep acquisition running (see `measure` for details). "
                "A 9.9E37 result is the scope's invalid/overflow sentinel and comes back annotated; "
                "any source channel whose display is OFF is auto-enabled first (noted in the result). "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source1": {"type": "string", "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"], "description": "Reference channel"},
                    "source2": {"type": "string", "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"], "description": "Measured channel"},
                    "item":    {"type": "string", "enum": [
                        "RDELAY", "FDELAY", "RPHASE", "FPHASE",
                        "RRDELAY", "RFDELAY", "FRDELAY", "FFDELAY",
                        "RRPHASE", "RFPHASE", "FRPHASE", "FFPHASE",
                    ]},
                },
                "required": ["source1", "source2", "item"],
            },
        ),
        types.Tool(
            name="get_waveform",
            description=(
                "Download and analyse the current waveform for a channel (NORM screen buffer, up to ~1000–1200 points depending on scope). "
                "Preferred over screenshot for inspecting the trace — the text analysis is "
                "cheaper and easier to reason over than an image. "
                "Stop or single-trigger the scope first for consistent data. "
                "By default returns a plain-text analysis: signal shape, frequency/period, amplitude, "
                "DC offset, cycle count, and data-quality warnings (e.g. mid-cycle edges, invalid frequency). "
                "Amplitude is judged against the channel's V/div: a trace filling under ~10% of the vertical "
                "screen is flagged as noise floor and its shape/frequency are not reported, and one filling "
                "under ~20% gets a low-amplitude warning (reduce V/div and re-capture for a clean signal). "
                "Set raw_data=true to get screen time/voltage JSON arrays instead; this flag "
                "only controls representation, not RAW acquisition memory. Use export_acquisition "
                "for the complete stopped record. "
                "If the channel's display is OFF it is auto-enabled first (flagged in the warnings). "
                "After reading, act on any warnings — if FREQUENCY would be 9.9E37 widen the timebase; "
                "if edges are not near the DC mean, adjust offset so right edge = N×(period/2) − 6×scale. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "channel":  {"type": "string", "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"]},
                    "raw_data": {"type": "boolean", "description": "Return raw time/voltage JSON arrays instead of text analysis (default false)"},
                },
                "required": ["channel"],
            },
        ),
        types.Tool(
            name="set_cursors",
            description=(
                "Set cursor mode and/or X positions. "
                "mode: OFF, MANUAL (fixed time positions, reads voltage at those X points), "
                "TRACK (cursors snap to and follow the waveform at the X position). "
                "Omit mode to keep current mode. "
                "ax/bx: cursor A/B time positions in seconds. "
                "Returns the resulting cursor readouts. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["OFF", "MANUAL", "TRACK"]},
                    "ax":   {"type": ["number", "string"], "description": "Cursor A X position in seconds"},
                    "bx":   {"type": ["number", "string"], "description": "Cursor B X position in seconds"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_cursor_values",
            description=(
                "Read current cursor mode and all cursor readouts. "
                "AX_s and BX_s are time positions in seconds. "
                "inv_delta_x is 1/Δt — the frequency between the two cursors. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="send_raw",
            description=(
                "Send an arbitrary SCPI command. "
                "Queries (ending with '?') return the response string; "
                "writes return empty string and auto-check the error queue. "
                "Use as an escape hatch when no dedicated tool covers the operation. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "SCPI command, e.g. ':CHAN1:SCAL?' or ':CHAN1:DISP ON'"},
                },
                "required": ["command"],
            },
        ),
        types.Tool(
            name="check_error",
            description="Query the SCPI error queue. Returns the error if present, or 'No error' if clear. Do not call concurrently with any other rigol tool.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="run",
            description="Start continuous acquisition. Returns trigger status after the command. Do not call concurrently with any other rigol tool.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="stop",
            description=(
                "Stop acquisition and freeze the display. "
                "Use before reading measurements or cursors for stable values. "
                "Returns trigger status after the command. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="single",
            description=(
                "Arm the scope for a single acquisition; stops automatically after one trigger event. "
                "Returns trigger status. "
                "Note: acquisition does not complete until a trigger occurs — "
                "call stop or check trigger status before reading measurements. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="autoscale",
            description=(
                "Run the scope's auto-setup (timebase, vertical scale, trigger). "
                "Takes a few seconds; call get_scope_state afterwards to see the resulting configuration. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="decode_spi",
            description=(
                "Decode SPI from the last acquisition and return one line per frame: start time, "
                "bit count, and MOSI/MISO in hex. The channels are downloaded as samples and decoded "
                "in software, so the scope's own decoder does not need to be set up. "
                "Single-trigger first (e.g. on chip select): a stopped scope is read from acquisition "
                "memory, a running one from the screen buffer only. "
                "With cs, frames are chip-select spans; without it, frames split at clock gaps. "
                "Thresholds default to the midpoint of each channel's swing. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sck":  {"type": "string", "enum": _CHANNELS, "description": "Clock channel"},
                    "cs":   {"type": "string", "enum": _CHANNELS, "description": "Chip-select channel (optional)"},
                    "mosi": {"type": "string", "enum": _CHANNELS, "description": "MOSI channel (optional)"},
                    "miso": {"type": "string", "enum": _CHANNELS, "description": "MISO channel (optional)"},
                    "mode": {"type": "integer", "enum": [0, 1, 2, 3], "description": "SPI mode (default 0)"},
                    "cs_active_low": {"type": "boolean", "description": "Default true"},
                    "msb_first": {"type": "boolean", "description": "Default true"},
                    "threshold_v": {"type": ["number", "string"], "description": "Logic threshold for every channel (default: per-channel midpoint)"},
                    "max_frames": {"type": "integer", "description": "Frames to list (default 64)"},
                },
                "required": ["sck"],
            },
        ),
    ]
    # send_raw is an arbitrary-SCPI escape hatch — only expose it when explicitly enabled.
    if not _send_raw_enabled():
        tools = [t for t in tools if t.name != "send_raw"]
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.ContentBlock]:
    global _last_call_time
    if name == "disconnect":
        # Deliberately avoid _call: it would open a scope just to disconnect it.
        async with _scope_lock:
            result = disconnect_scope()
            _last_call_time = time.monotonic()
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    if name == "get_acquisition_info":
        result = await _call(get_acquisition_info)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    if name == "export_acquisition":
        unknown = set(arguments) - {"channels", "output_dir"}
        if unknown:
            raise ValueError(f"Unsupported export arguments: {sorted(unknown)}; only complete records supported")
        result = await _call(export_acquisition, channels=arguments.get("channels"),
                             output_dir=arguments.get("output_dir"), _retry=False)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    if name == "screenshot":
        png_bytes = await _call(screenshot_png)
        # pyvisa-py (@py, used with the WinUSB driver) can leave the USBTMC stream in a
        # state where the command right after a large screenshot transfer intermittently
        # times out; dropping the connection forces a clean USBTMC session on the next
        # call (~1 s, within the recovery window below). NI-VISA (@ivi, native USBTMC
        # driver) recovers on its own and a reconnect there actually disrupts the next
        # command — so reconnect only on @py.
        if usb_in_use() and active_backend() == "@py":
            async with _scope_lock:
                invalidate_scope()
        # Advance the cooldown timestamp so the next _call waits for the scope to recover
        # after the large display transfer before sending further SCPI commands.
        _last_call_time = time.monotonic() + _POST_SCREENSHOT_DELAY

        save_dir = Path(os.environ.get("RIGOL_SCREENSHOT_DIR", "screenshots")).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        filename.write_bytes(png_bytes)

        b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        return [
            types.TextContent(type="text", text=f"Saved: {filename}"),
            types.ImageContent(type="image", data=b64, mimeType="image/png"),
        ]

    if name == "idn":
        # connection_info reads env vars + module-level state with no device I/O. It's
        # safe to call after any outcome: on success we show what got selected (driver,
        # resource, backend), on failure we show what was attempted (transport, env vars).
        # That way the diagnostic surfaces config issues (LAN vs USB, wrong IP) that
        # would otherwise be hidden behind an opaque VI_ERROR_TMO timeout.
        try:
            idn_str = await _call(idn)
            # Populate the driver cache from the IDN we already have, so the diagnostic
            # below shows which dialect was selected. Avoids a second *IDN? round-trip
            # that get_driver(scope) would otherwise do on first dialect use.
            set_driver_from_idn(idn_str)
            info = connection_info()
            diag = "\n".join(f"  {k:18s}: {v}" for k, v in info.items())
            return [types.TextContent(type="text",
                text=f"Connection:\n{diag}\n\nIDN: {idn_str}")]
        except Exception as exc:
            info = connection_info()
            diag = "\n".join(f"  {k:18s}: {v}" for k, v in info.items())
            return [types.TextContent(type="text",
                text=f"Connection:\n{diag}\n\n"
                     f"*IDN? query FAILED: {type(exc).__name__}: {exc}\n\n"
                     "The connection details above show what the server was attempting "
                     "when the query failed — check that transport (USB/LAN), RIGOL_IP, "
                     "and RIGOL_USB match how the scope is actually connected.")]

    if name == "get_scope_state":
        state = await _call(get_scope_state)
        return [types.TextContent(type="text", text=json.dumps(state, indent=2))]

    if name == "set_channel":
        def _f(key):
            v = arguments.get(key)
            return float(v) if v is not None else None

        await _call(
            set_channel,
            arguments["channel"],
            display=arguments.get("display"),
            scale=_f("scale_v_div"),
            offset=_f("offset_v"),
            coupling=arguments.get("coupling"),
            probe=_f("probe"),
        )
        state = await _call(get_scope_state)
        ch = arguments["channel"].upper()
        return [types.TextContent(type="text", text=json.dumps(state["channels"][ch], indent=2))]

    if name == "set_timebase":
        def _f(key):
            v = arguments.get(key)
            return float(v) if v is not None else None

        await _call(set_timebase, scale=_f("scale_s_div"), offset=_f("offset_s"))
        state = await _call(get_scope_state)
        return [types.TextContent(type="text", text=json.dumps(state["timebase"], indent=2))]

    if name == "set_trigger":
        level = arguments.get("level")
        await _call(
            set_trigger,
            source=arguments.get("source"),
            slope=arguments.get("slope"),
            level=float(level) if level is not None else None,
        )
        state = await _call(get_scope_state)
        return [types.TextContent(type="text", text=json.dumps(state["trigger"], indent=2))]

    if name == "measure":
        value = await _call(measure, arguments["channel"], arguments["item"])
        return [types.TextContent(
            type="text",
            text=f"{arguments['item']} on {arguments['channel']}: {value}",
        )]

    if name == "measure_between":
        value = await _call(measure_between, arguments["source1"], arguments["source2"], arguments["item"])
        return [types.TextContent(
            type="text",
            text=f"{arguments['item']} from {arguments['source1']} to {arguments['source2']}: {value}",
        )]

    if name == "get_waveform":
        data = await _call(get_waveform, arguments["channel"])
        if arguments.get("raw_data"):
            return [types.TextContent(type="text", text=json.dumps(data))]
        return [types.TextContent(type="text", text=_describe_waveform(data))]

    if name == "set_cursors":
        mode = arguments.get("mode")
        ax = float(arguments["ax"]) if "ax" in arguments else None
        bx = float(arguments["bx"]) if "bx" in arguments else None
        if mode is not None:
            await _call(set_cursor_mode, mode)
        else:
            mode = await _call(get_cursor_mode)
        if mode.upper() != "OFF" and (ax is not None or bx is not None):
            await _call(set_cursor_positions, mode, ax=ax, bx=bx)
        values = await _call(get_cursor_values)
        lines = "\n".join(f"{k}: {v}" for k, v in values.items())
        return [types.TextContent(type="text", text=lines)]

    if name == "get_cursor_values":
        values = await _call(get_cursor_values)
        lines = "\n".join(f"{k}: {v}" for k, v in values.items())
        return [types.TextContent(type="text", text=lines)]

    if name == "send_raw":
        if not _send_raw_enabled():
            raise ValueError(
                f"send_raw is disabled. Set {_SEND_RAW_ENV}=1 to enable arbitrary SCPI commands."
            )
        response = await _call(send_raw, arguments["command"])
        return [types.TextContent(type="text", text=response or "(no response)")]

    if name == "check_error":
        err = await _call(check_scpi_error)
        return [types.TextContent(type="text", text=err or "No error")]

    if name in ("run", "stop", "single"):
        fn = {"run": run, "stop": stop, "single": single}[name]
        status = await _call(fn)
        return [types.TextContent(type="text", text=f"trigger status: {status}")]

    if name == "autoscale":
        await _call(autoscale)
        state = await _call(get_scope_state)
        return [types.TextContent(type="text", text=json.dumps(state, indent=2))]

    if name == "decode_spi":
        return [types.TextContent(type="text", text=await _decode_spi(arguments))]

    raise ValueError(f"Unknown tool: {name}")


def _trigger_status(scope) -> str:
    return scope.query(":TRIGger:STATus?").strip()


async def _decode_spi(arguments: dict) -> str:
    roles = {r: arguments[r].upper() for r in ("sck", "cs", "mosi", "miso") if arguments.get(r)}
    raw = (await _call(_trigger_status)) == "STOP"
    captured = {}
    for ch in dict.fromkeys(roles.values()):
        captured[ch] = await _call(get_samples, ch, raw)
    first = captured[roles["sck"]]
    t0, dt = first["t0_s"], first["dt_s"]
    n = min(len(c["volts"]) for c in captured.values())
    if n == 0:
        raise RuntimeError("No samples returned. Single-trigger the scope and retry.")
    fixed = arguments.get("threshold_v")
    swing = {}
    digital = {}
    for ch, c in captured.items():
        volts = c["volts"][:n]
        lo, hi = min(volts), max(volts)
        threshold = float(fixed) if fixed is not None else (lo + hi) / 2
        swing[ch] = (lo, hi, threshold)
        digital[ch] = spi_decode.digitize(volts, threshold, max((hi - lo) * 0.1, 0.05))

    line = lambda role: digital[roles[role]] if role in roles else None
    mode = int(arguments.get("mode", 0))
    frames = spi_decode.decode_spi(
        t0, dt, line("sck"), line("mosi"), line("miso"), line("cs"),
        mode=mode, cs_active_low=arguments.get("cs_active_low", True),
    )
    msb_first = arguments.get("msb_first", True)
    freq = spi_decode.clock_frequency(line("sck"), dt)

    out = [
        f"SPI mode {mode}, {'acquisition memory' if raw else 'screen buffer (scope running)'}, "
        f"{n} samples at {dt * 1e9:.3g} ns, SCK ~{freq / 1e6:.3g} MHz" if freq else
        f"SPI mode {mode}, {n} samples at {dt * 1e9:.3g} ns, no clock edges on {roles['sck']}",
    ]
    for role, ch in roles.items():
        lo, hi, th = swing[ch]
        out.append(f"  {role.upper()} {ch}: {lo:.2f} V to {hi:.2f} V, threshold {th:.2f} V")
    warnings = []
    if freq and 1 / (freq * dt) < 4:
        warnings.append("fewer than 4 samples per clock period; shorten the timebase or raise the memory depth")
    for role, ch in roles.items():
        lo, hi, _ = swing[ch]
        if hi - lo < 1.0:
            warnings.append(f"{role.upper()} {ch} swings only {hi - lo:.2f} V; it may not be connected or driven")
    max_frames = int(arguments.get("max_frames", 64))
    out.append(f"{len(frames)} frames")
    for i, f in enumerate(frames[:max_frames]):
        parts = [f"{i + 1:3d}  t={f.start_s * 1e6:9.3f} us  {f.bit_count:3d} bits"]
        if f.mosi_bits:
            parts.append(f"MOSI {spi_decode.bits_to_hex(f.mosi_bits, msb_first)}")
        if f.miso_bits:
            parts.append(f"MISO {spi_decode.bits_to_hex(f.miso_bits, msb_first)}")
        if f.truncated:
            parts.append("(cut off by the capture window)")
        out.append("  ".join(parts))
    if len(frames) > max_frames:
        out.append(f"... {len(frames) - max_frames} more")
    if warnings:
        out.append("Warnings:")
        out.extend(f"  {w}" for w in warnings)
    return "\n".join(out)


def main() -> None:
    asyncio.run(_run())


async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    main()
