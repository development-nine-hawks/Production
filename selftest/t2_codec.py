"""Layer 2 — payload codec. EXHAUSTIVE over the whole 2^16 seed space."""
import random

from _harness import check, info, need, import_engine, run_module, finish

eng, _ = import_engine()

split      = need(eng, "split_seed_for_dm")
split_v0   = need(eng, "split_seed_for_dm_legacy")
recombine  = need(eng, "recombine_seed_from_dm")
m_mul      = need(eng, "_m4_mul")
m_pack     = need(eng, "_m4_pack")
m_unpack   = need(eng, "_m4_unpack")
b32_decode = need(eng, "_b32_decode")
b32_encode = need(eng, "_b32_encode")
check_char = need(eng, "_check_char")
B32_ALPHA  = need(eng, "_B32_ALPHA")
SEED_MASK  = need(eng, "SEED_MASK")
V_MATRIX   = need(eng, "_DM_VERSION_MATRIX")

ALL_SEEDS = range(1 << 16)


def test_round_trip_exhaustive():
    """Every seed that can exist. Not a sample — the entire domain."""
    bad = [s for s in ALL_SEEDS if recombine(*split(s)) != s]
    check(not bad, "split -> recombine round-trips for ALL 65,536 seeds",
          f"{len(bad)} failures, first: {bad[:5]}")


def test_the_seed_really_is_the_matrix_product():
    """The whole point of this design, asserted directly against the payload.

    Pull A and B straight out of the printed characters and multiply them by
    hand; the result must be the seed.
    """
    bad = []
    for s in range(0, 1 << 16, 7):
        a, b = split(s)
        field35 = b32_decode((a + b)[:7])
        A = m_unpack((field35 >> 16) & 0xFFFF)
        B = m_unpack(field35 & 0xFFFF)
        if m_pack(m_mul(A, B)) != s:
            bad.append((s, A, B))
    check(not bad, "A · B == seed, computed directly from the printed payload "
                   "(9,363 seeds checked)",
          f"{len(bad)} mismatches, first: {bad[:2]}")

    s = 0x1234
    a, b = split(s)
    f = b32_decode((a + b)[:7])
    A, B = m_unpack((f >> 16) & 0xFFFF), m_unpack(f & 0xFFFF)
    info(f"worked example: seed 0x{s:04X} -> codes '{a}' and '{b}'")
    info(f"  A = {A}")
    info(f"  B = {B}")
    info(f"  A · B = {m_mul(A, B)} = 0x{m_pack(m_mul(A, B)):04X}")


def test_every_seed_maps_to_a_distinct_pair():
    pairs = {split(s) for s in ALL_SEEDS}
    check(len(pairs) == (1 << 16),
          f"all 65,536 seeds produce distinct code pairs ({len(pairs):,})",
          f"only {len(pairs):,} distinct — two patterns would share one label")


def test_shares_are_well_formed():
    bad = []
    for s in range(0, 1 << 16, 3):
        a, b = split(s)
        if len(a) != 4 or len(b) != 4 or not all(c in B32_ALPHA for c in a + b):
            bad.append((s, a, b))
    check(not bad, "every share is exactly 4 Base32 characters",
          f"{len(bad)} malformed, first: {bad[:3]}")


def test_shares_are_never_identical():
    """share_a == share_b makes a label PERMANENTLY unverifiable: both dedup
    paths downstream key on decoded text, so the pair collapses to one string
    and can never be re-formed."""
    same = [s for s in ALL_SEEDS if (p := split(s))[0] == p[1]]
    check(not same, "no seed produces two identical codes (all 65,536 checked)",
          f"{len(same)} self-colliding seeds: {same[:5]}")


def test_seed_is_masked_not_silently_truncated():
    """A caller passing a wider seed must still get a label that verifies."""
    bad = []
    for wide in (0x1_0000, 0xDEADBEEF, 0x7FFFFFFF, 1 << 30):
        got = recombine(*split(wide))
        if got != (wide & SEED_MASK):
            bad.append((hex(wide), got, hex(wide & SEED_MASK)))
    check(not bad, "an over-wide seed is masked consistently on both sides, so "
                   "the label still round-trips",
          f"{bad}")


def test_version_separates_v0_from_v2():
    v2 = {split(s)[0][0] for s in range(0, 1 << 16, 11)}
    v0 = {split_v0(s)[0][0] for s in range(0, 1 << 20, 37)}
    info(f"v2 (matrix) leading chars: {sorted(v2)}")
    info(f"v0 (legacy) leading chars: {sorted(v0)}")
    check(not (v2 & v0), "v0 and v2 payloads never share a leading character, "
                         "so the two formats can always be told apart",
          f"overlap: {sorted(v2 & v0)}")


def test_unknown_versions_rejected():
    r = random.Random(1)
    accepted = []
    for _ in range(20_000):
        ver = r.choice([1, 3, 4, 5, 6, 7])       # 0 and 2 are the real ones
        p7 = b32_encode((ver << 32) | r.getrandbits(32), 7)
        full8 = p7 + check_char(p7)
        if recombine(full8[:4], full8[4:]) is not None:
            accepted.append((ver, full8))
    check(not accepted, "payloads tagged with unallocated versions are rejected",
          f"{len(accepted)} accepted, first: {accepted[:3]}")


def test_malformed_input_returns_none_never_raises():
    r = random.Random(2)
    junk = ["", "A", "AAA", "AAAAA", "aaaa", "1234", "!!!!", " IU7I", None,
            "IU7I" * 3, "\x00\x00\x00\x00", "0000", "ZZZZ"]
    junk += ["".join(r.choice(B32_ALPHA + "abc!@# ") for _ in range(r.randint(0, 9)))
             for _ in range(15_000)]
    problems = []
    for a in junk:
        for b in junk[:40]:
            try:
                res = recombine(a, b)
                if res is not None and not isinstance(res, int):
                    problems.append(("bad type", a, b, res))
            except Exception as e:
                problems.append((type(e).__name__, repr(a), repr(b), str(e)))
    check(not problems, "recombine_seed_from_dm never raises and returns None "
                        "or an int across fuzzed input",
          f"{len(problems)} problems, first: {problems[:3]}")


if __name__ == "__main__":
    run_module(globals(), "LAYER 2 — payload codec (exhaustive over 2^16)")
    finish()
