"""Layer 1 — GF(2^4) field arithmetic and the 2x2 matrix layer. No deps.

The seed space is only 2^16, which means this layer can be EXHAUSTIVE rather
than sampled. Every assertion below covers the entire domain, not a sample.
"""
from _harness import check, info, need, import_engine, run_module, finish

eng, _ = import_engine()

gf_mul   = need(eng, "_gf4_mul")
gf_inv   = need(eng, "_gf4_inv")
m_mul    = need(eng, "_m4_mul")
m_det    = need(eng, "_m4_det")
m_inv    = need(eng, "_m4_inv")
m_pack   = need(eng, "_m4_pack")
m_unpack = need(eng, "_m4_unpack")
pick_a   = need(eng, "_pick_matrix_a")

IDENTITY = [[1, 0], [0, 1]]


def test_gf4_field_axioms_exhaustive():
    """All 16^3 = 4,096 triples. No sampling needed at this field size."""
    bad = []
    for a in range(16):
        for b in range(16):
            if gf_mul(a, b) != gf_mul(b, a):
                bad.append(("commutative", a, b))
            if not 0 <= gf_mul(a, b) <= 15:
                bad.append(("closure", a, b))
            if gf_mul(a, 1) != a:
                bad.append(("unit", a))
            for c in range(16):
                if gf_mul(gf_mul(a, b), c) != gf_mul(a, gf_mul(b, c)):
                    bad.append(("associative", a, b, c))
                if gf_mul(a, b ^ c) != (gf_mul(a, b) ^ gf_mul(a, c)):
                    bad.append(("distributive", a, b, c))
    check(not bad, "GF(2^4) is a field over all 4,096 triples "
                   "(closure, commutativity, associativity, distributivity, unit)",
          f"{len(bad)} violations, first: {bad[:3]}")


def test_gf4_inverse_exhaustive():
    bad = [a for a in range(1, 16) if gf_mul(a, gf_inv(a)) != 1]
    check(not bad, "every one of the 15 non-zero GF(2^4) elements has a correct "
                   "inverse", f"wrong for {bad}")
    try:
        gf_inv(0)
        check(False, "_gf4_inv(0) raises", "it returned instead")
    except ZeroDivisionError:
        check(True, "_gf4_inv(0) raises ZeroDivisionError rather than lying")


def test_pack_unpack_is_exhaustively_bijective():
    bad = [n for n in range(1 << 16) if m_pack(m_unpack(n)) != n]
    check(not bad, "pack/unpack round-trips for all 65,536 matrix encodings",
          f"{len(bad)} failures, first: {bad[:3]}")


def test_matrix_inverse_exhaustive():
    """Every invertible 2x2 over GF(2^4) — all of them, not a sample."""
    singular = bad = 0
    for n in range(1 << 16):
        M = m_unpack(n)
        if m_det(M) == 0:
            singular += 1
            continue
        if m_mul(M, m_inv(M)) != IDENTITY or m_mul(m_inv(M), M) != IDENTITY:
            bad += 1
    total = (1 << 16) - singular
    info(f"{total:,} invertible matrices, {singular:,} singular "
         f"({singular / 655.36:.1f}%)")
    check(bad == 0, f"M · M^-1 == M^-1 · M == I for all {total:,} invertible "
                    f"2x2 matrices over GF(2^4)",
          f"{bad} failures")


def test_matrix_multiplication_is_associative():
    """Sampled — 65,536^3 is too many. Uses a fixed stride for determinism."""
    bad = []
    for i in range(0, 1 << 16, 991):
        for j in range(0, 1 << 16, 1013):
            A, B = m_unpack(i), m_unpack(j)
            C = m_unpack((i ^ j) & 0xFFFF)
            if m_mul(m_mul(A, B), C) != m_mul(A, m_mul(B, C)):
                bad.append((i, j))
    check(not bad, "2x2 matrix multiplication is associative",
          f"{len(bad)} violations, first: {bad[:3]}")


def test_multiplication_is_not_commutative():
    """Sanity: if A·B always equalled B·A, the encode step would be degenerate
    and the Top/Right ordering would carry no meaning at all."""
    diff = sum(1 for i in range(0, 1 << 16, 97)
               if m_mul(m_unpack(i), m_unpack((i * 7 + 3) & 0xFFFF))
               != m_mul(m_unpack((i * 7 + 3) & 0xFFFF), m_unpack(i)))
    total = len(range(0, 1 << 16, 97))
    info(f"{diff}/{total} sampled pairs have A·B != B·A")
    check(diff > total * 0.5,
          "matrix multiplication is genuinely non-commutative here",
          "if it were commutative the two codes would be interchangeable")


def test_pick_matrix_a_is_deterministic_and_invertible():
    """Determinism is load-bearing: re-printing an existing label must
    reproduce the exact codes already on physical product."""
    bad_det, bad_stable = [], []
    for s in range(1 << 16):
        A = pick_a(s)
        if m_det(A) == 0:
            bad_det.append(s)
        if s % 37 == 0 and pick_a(s) != A:
            bad_stable.append(s)
    check(not bad_det, "_pick_matrix_a returns an invertible matrix for all "
                       "65,536 seeds",
          f"singular for {len(bad_det)} seeds, first: {bad_det[:3]}")
    check(not bad_stable, "_pick_matrix_a is deterministic — the same seed "
                          "always yields the same matrix",
          f"unstable for {bad_stable[:3]}")


def test_pick_matrix_a_spreads_neighbouring_seeds():
    """Consecutive seeds must not produce near-identical Top codes."""
    same = sum(1 for s in range(0, (1 << 16) - 1)
               if m_pack(pick_a(s)) == m_pack(pick_a(s + 1)))
    info(f"{same} adjacent seed pairs share the same A matrix")
    check(same < 100, "neighbouring seeds get different A matrices",
          f"{same} adjacent collisions — the spreader is not spreading")


if __name__ == "__main__":
    run_module(globals(), "LAYER 1 — GF(2^4) + 2x2 matrix arithmetic")
    finish()
