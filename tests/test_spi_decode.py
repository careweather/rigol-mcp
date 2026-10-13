from rigol_mcp import spi_decode
from rigol_mcp import scope as sc
from tests.conftest import FakeScope, make_block


def _mode0_frame(mosi_word: int, miso_word: int, bits: int, samples_per_half: int = 4):
    """Lines for one mode-0 frame: CS low, data set while SCK is low, sampled on rising SCK."""
    sck, cs, mosi, miso = [0] * 8, [1] * 8, [0] * 8, [0] * 8
    for b in range(bits - 1, -1, -1):
        mo, mi = (mosi_word >> b) & 1, (miso_word >> b) & 1
        for level in (0, 1):
            sck += [level] * samples_per_half
            cs += [0] * samples_per_half
            mosi += [mo] * samples_per_half
            miso += [mi] * samples_per_half
    sck += [0] * 8
    cs += [0] * 2 + [1] * 6
    mosi += [0] * 8
    miso += [0] * 8
    return sck, cs, mosi, miso


def test_decode_mode0_frame_with_chip_select():
    sck, cs, mosi, miso = _mode0_frame(0xA5C30F, 0x5A3CF0, 24)
    frames = spi_decode.decode_spi(0.0, 1e-9, sck, mosi, miso, cs)
    assert len(frames) == 1
    assert frames[0].bit_count == 24
    assert spi_decode.bits_to_hex(frames[0].mosi_bits) == "A5C30F"
    assert spi_decode.bits_to_hex(frames[0].miso_bits) == "5A3CF0"
    assert not frames[0].truncated


def test_decode_splits_at_clock_gaps_without_chip_select():
    a = _mode0_frame(0xA5, 0x00, 8)
    b = _mode0_frame(0x3C, 0x00, 8)
    sck = a[0] + [0] * 200 + b[0]
    mosi = a[2] + [0] * 200 + b[2]
    frames = spi_decode.decode_spi(0.0, 1e-9, sck, mosi, None)
    assert [spi_decode.bits_to_hex(f.mosi_bits) for f in frames] == ["A5", "3C"]


def test_digitize_hysteresis_ignores_ringing():
    volts = [0.0, 3.3, 1.7, 1.6, 1.7, 3.3, 0.0]
    assert spi_decode.digitize(volts, 1.65, 0.3) == [0, 1, 1, 1, 1, 1, 0]


def test_bits_to_hex_partial_nibble():
    assert spi_decode.bits_to_hex([1, 0, 1, 0, 1, 1]) == "A:11"


def test_get_samples_reads_bytes_in_chunks(monkeypatch):
    monkeypatch.setattr(sc, "_RAW_CHUNK_POINTS", 3)
    pre = "0,2,5,1,1.0e-08,-1.0e-06,0,0.1,0,127"
    s = FakeScope(responses={":WAV:PRE?": pre})
    s.load(make_block(bytes([127, 137, 147])) + make_block(bytes([117, 127])))
    data = sc.get_samples(s, "chan2", raw=True)
    assert ":WAV:MODE RAW" in s.written and ":WAV:FORM BYTE" in s.written
    assert [":WAV:STAR 1", ":WAV:STOP 3"] == s.written[3:5]
    assert [round(v, 3) for v in data["volts"]] == [0.0, 1.0, 2.0, -1.0, 0.0]
    assert data["t0_s"] == -1.0e-06
    assert data["dt_s"] == 1.0e-08
