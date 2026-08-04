"""Layer 3 — the Top-vs-Right classifier invariant. EXHAUSTIVE.

recombine_seed_from_dm is not only the decoder. Four call sites use its
None/not-None result to decide which DataMatrix is Top and which is Right,
rotation-invariantly. When it accepts a SWAPPED pair nothing errors: the crop
is fitted to transposed correspondences and produces a confident wrong verdict.

Note this is the reason the check character survived the redesign. A pure
matrix product carries no redundancy — every 8-character pair would multiply to
*some* seed, so there would be nothing to reject a misread code with, and no
signal at all for orientation.
"""
import random

from _harness import check, info, need, import_engine, run_module, finish

eng, _ = import_engine()
import _v0_codec as v0

split     = need(eng, "split_seed_for_dm")
recombine = need(eng, "recombine_seed_from_dm")
B32_ALPHA = need(eng, "_B32_ALPHA")

SWAP_BUDGET = 0.025     # see the note in test_swapped_order_false_accept_rate


def test_correct_order_always_decodes_exhaustive():
    """Zero tolerance. One miss is a label that can never be verified."""
    bad = [s for s in range(1 << 16) if recombine(*split(s)) != s]
    check(not bad, "correctly-ordered pairs decode for ALL 65,536 seeds",
          f"{len(bad)} failures, first: {bad[:5]}")


def test_swapped_order_false_accept_rate():
    """Measured over the entire seed space, and compared against the codec
    this branch replaced so the number means something.

    The pre-existing baseline is 1/16, not the 1/32 the design implies. That is
    structural: substituting the check-char definition into the swapped-order
    constraint leaves all-even coefficients, so it can only hit the 16 even
    residues, and all-odd check weights (required for the 100% single-char
    substitution property) make that unavoidable.

    The 3-bit version tag mitigates it — a swapped pair's leading character is
    uniform over all 32 symbols, so it must also land on a recognised version,
    which is 2 chances in 8. Expected 1/16 * 1/4 = 1/64 = 1.56%.
    """
    n = 1 << 16
    swapped = sum(1 for s in range(n)
                  if recombine(*reversed(split(s))) is not None)
    rate = swapped / n
    info(f"swapped-order accept: {swapped:,}/{n:,} = {rate * 100:.3f}%")

    r0 = random.Random(21)
    m = 100_000
    old = sum(1 for _ in range(m)
              if v0.recombine_seed_from_dm(
                  *reversed(v0.split_seed_for_dm(r0.getrandbits(32)))) is not None)
    info(f"same measurement on the codec this replaced: {old / m * 100:.3f}%")

    check(rate <= SWAP_BUDGET,
          f"swapped-pair false-accept {rate * 100:.2f}% is within the "
          f"{SWAP_BUDGET * 100:.1f}% budget",
          "orientation ambiguity regressed; every ambiguous label is a "
          "coin-flip on crop geometry")
    check(rate < old / m,
          f"the version tag improves orientation robustness "
          f"({old / m * 100:.2f}% -> {rate * 100:.2f}%)",
          "no improvement over the old codec — is the version check active?")


def test_random_pairs_are_rejected():
    r = random.Random(22)
    n = 200_000
    acc = sum(1 for _ in range(n)
              if recombine("".join(r.choice(B32_ALPHA) for _ in range(4)),
                           "".join(r.choice(B32_ALPHA) for _ in range(4)))
              is not None)
    rate = acc / n
    info(f"random 4+4 pair accept: {acc:,}/{n:,} = {rate * 100:.3f}% "
         f"(check char alone = 1/32 = 3.13%; with the version tag "
         f"1/32 * 2/8 = 0.78%)")
    check(rate < 0.02, f"unrelated random pairs rejected {(1 - rate) * 100:.2f}% "
                       f"of the time", f"accept rate {rate * 100:.2f}%")


def test_cross_label_pairs_no_worse_than_before():
    """Two codes from two genuine labels — e.g. two products in one photo.

    The version tag cannot help here: share_a comes from a genuine v2 label, so
    its version bits are correct and only the check character stands in the
    way, giving the full 1/32. That is identical to the pre-change behaviour,
    so this asserts NO REGRESSION rather than an invented absolute threshold.
    """
    r = random.Random(23)
    n = 100_000
    new = sum(1 for _ in range(n)
              if recombine(split(r.getrandbits(16))[0],
                           split(r.getrandbits(16))[1]) is not None)
    r = random.Random(23)
    old = sum(1 for _ in range(n)
              if v0.recombine_seed_from_dm(
                  v0.split_seed_for_dm(r.getrandbits(32))[0],
                  v0.split_seed_for_dm(r.getrandbits(32))[1]) is not None)
    info(f"cross-label accept — before: {old / n * 100:.3f}%  "
         f"after: {new / n * 100:.3f}%  (1/32 = {100 / 32:.3f}%)")
    check(new / n <= old / n + 0.005,
          f"cross-label false-accept ({new / n * 100:.2f}%) is no worse than "
          f"the pre-change baseline ({old / n * 100:.2f}%)",
          "two labels in one photo got measurably worse")


def test_classifier_is_deterministic():
    bad = []
    for s in range(0, 1 << 16, 101):
        a, b = split(s)
        if len({recombine(a, b) for _ in range(5)}) != 1:
            bad.append(s)
    check(not bad, "recombine is deterministic across repeated calls",
          f"{len(bad)} non-deterministic")


if __name__ == "__main__":
    run_module(globals(), "LAYER 3 — Top-vs-Right classifier invariant")
    finish()
