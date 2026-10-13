"""Software SPI decoder over captured channel samples.

The DS1000Z decodes SPI on screen but has no SCPI query for the decoded words, so the
channels are downloaded and decoded here. Every channel is sampled on the same time base,
so sample index i is the same instant on every line.
"""

from dataclasses import dataclass, field


@dataclass
class SpiFrame:
    start_s: float
    end_s: float
    mosi_bits: list[int] = field(default_factory=list)
    miso_bits: list[int] = field(default_factory=list)
    truncated: bool = False  # chip select was already active at the start or still active at the end

    @property
    def bit_count(self) -> int:
        return max(len(self.mosi_bits), len(self.miso_bits))


def digitize(volts: list[float], threshold: float, hysteresis: float) -> list[int]:
    """Threshold with hysteresis so ringing near the threshold does not add edges."""
    high = threshold + hysteresis
    low = threshold - hysteresis
    state = 1 if volts and volts[0] >= threshold else 0
    out = []
    for v in volts:
        if state and v < low:
            state = 0
        elif not state and v > high:
            state = 1
        out.append(state)
    return out


def default_threshold(volts: list[float]) -> float:
    return (min(volts) + max(volts)) / 2 if volts else 0.0


def _active_spans(cs: list[int], active_level: int) -> list[tuple[int, int, bool]]:
    """(first, last, truncated) index spans where chip select is at its active level."""
    spans = []
    start = None
    for i, level in enumerate(cs):
        if level == active_level and start is None:
            start = i
        elif level != active_level and start is not None:
            spans.append((start, i - 1, start == 0))
            start = None
    if start is not None:
        spans.append((start, len(cs) - 1, True))
    return spans


def _sample_edges(sck: list[int], rising: bool) -> list[int]:
    want_prev, want_now = (0, 1) if rising else (1, 0)
    return [i for i in range(1, len(sck)) if sck[i - 1] == want_prev and sck[i] == want_now]


def _gap_spans(edges: list[int], n: int) -> list[tuple[int, int, bool]]:
    """Without chip select, split the clock edges into frames at gaps of 4 clock periods."""
    if not edges:
        return []
    periods = sorted(b - a for a, b in zip(edges, edges[1:]))
    period = periods[len(periods) // 2] if periods else n
    spans = []
    first = edges[0]
    for a, b in zip(edges, edges[1:]):
        if b - a > 4 * period:
            spans.append((first, a, False))
            first = b
    spans.append((first, edges[-1], False))
    return spans


def decode_spi(
    t0: float,
    dt: float,
    sck: list[int],
    mosi: list[int] | None,
    miso: list[int] | None,
    cs: list[int] | None = None,
    mode: int = 0,
    cs_active_low: bool = True,
) -> list[SpiFrame]:
    """Decode digitized lines into frames. Data is sampled on the mode's capture edge.

    CPOL=0/CPHA=0 and CPOL=1/CPHA=1 capture on the rising edge, the other two on the falling edge.
    """
    cpol, cpha = (mode >> 1) & 1, mode & 1
    edges = _sample_edges(sck, rising=(cpol == cpha))
    spans = _active_spans(cs, 0 if cs_active_low else 1) if cs is not None else _gap_spans(edges, len(sck))
    frames = []
    edge_i = 0
    for first, last, truncated in spans:
        frame = SpiFrame(start_s=t0 + first * dt, end_s=t0 + last * dt, truncated=truncated or last == len(sck) - 1)
        while edge_i < len(edges) and edges[edge_i] < first:
            edge_i += 1
        while edge_i < len(edges) and edges[edge_i] <= last:
            i = edges[edge_i]
            if mosi is not None:
                frame.mosi_bits.append(mosi[i])
            if miso is not None:
                frame.miso_bits.append(miso[i])
            edge_i += 1
        if frame.bit_count:
            frames.append(frame)
    return frames


def bits_to_hex(bits: list[int], msb_first: bool = True) -> str:
    """Hex for whole nibbles. A trailing partial nibble is appended in binary after a colon."""
    if not msb_first:
        bits = bits[::-1]
    whole = len(bits) - len(bits) % 4
    text = "".join(f"{int(''.join(map(str, bits[i:i + 4])), 2):X}" for i in range(0, whole, 4))
    if len(bits) % 4:
        text += ":" + "".join(map(str, bits[whole:]))
    return text


def clock_frequency(sck: list[int], dt: float) -> float | None:
    rising = _sample_edges(sck, rising=True)
    periods = sorted(b - a for a, b in zip(rising, rising[1:]))
    if not periods:
        return None
    return 1.0 / (periods[len(periods) // 2] * dt)
