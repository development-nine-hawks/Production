"""FROZEN snapshot of the pre-HMAC (v0) codec, taken from branch `datamatrix`.

This file is a literal copy of cdp_engine.py lines 2358-2497 as they stood
BEFORE the HMAC + Hill change. It must NEVER import cdp_engine — the whole point
is that it survives independently, so t8_oldformat.py can generate genuine
old-format payloads and prove the new decoder handles them correctly.

If you find yourself "fixing" something in here, stop: a change means the
snapshot no longer represents what is printed on labels already in the field.
t8 asserts split_seed_for_dm(920789066) == ('AM3I','SC5Y') to catch exactly that.
"""

# Fixed 8-byte Feistel key  (4 × 16-bit round keys, concatenated)
_FEISTEL_KEY: bytes = bytes([0xA7, 0x3E, 0x2F, 0x91, 0xD8, 0x5C, 0x4B, 0xE6])

# Standard RFC 4648 Base32 alphabet  (uppercase A-Z  +  digits 2-7)
_B32_ALPHA: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'
_B32_IDX: dict = {c: i for i, c in enumerate(_B32_ALPHA)}

# Weights for the check character — all must be non-zero mod 32 (all are odd),
# ensuring that any single-character substitution changes the check value.
_CHECK_WEIGHTS: tuple = (3, 7, 11, 13, 17, 19, 23)


def _feistel_f(half: int, round_key: int) -> int:
    """Non-linear round function for the Feistel cipher (16-bit domain)."""
    v = (half ^ round_key) & 0xFFFF
    v = (v * 0x9E37 + 0xB5EF) & 0xFFFF
    v ^= v >> 7
    return (v * 0x6B5F) & 0xFFFF


def feistel_encrypt(seed: int) -> int:
    """
    4-round Feistel encryption of a 32-bit seed → 32-bit ciphertext.

    The cipher is a classic balanced Feistel network:
        L_{i+1} = R_i
        R_{i+1} = L_i XOR F(R_i, round_key_i)
    """
    L, R = (seed >> 16) & 0xFFFF, seed & 0xFFFF
    rks = [int.from_bytes(_FEISTEL_KEY[i:i+2], 'big') for i in range(0, 8, 2)]
    for rk in rks:
        L, R = R, L ^ _feistel_f(R, rk)
    return (L << 16) | R


def feistel_decrypt(ct: int) -> int:
    """
    4-round Feistel decryption of a 32-bit ciphertext → original seed.

    Inverse step (derived from the forward Feistel equations):
        R_i = L_{i+1}
        L_i = R_{i+1} XOR F(L_{i+1}, round_key_i)
    Applied in reverse key order.
    """
    L, R = (ct >> 16) & 0xFFFF, ct & 0xFFFF
    rks = [int.from_bytes(_FEISTEL_KEY[i:i+2], 'big') for i in range(0, 8, 2)]
    for rk in reversed(rks):
        L, R = R ^ _feistel_f(L, rk), L
    return (L << 16) | R


def _b32_encode(n: int, length: int = 7) -> str:
    """Encode integer n into `length` Base32 characters (big-endian, no padding)."""
    chars = []
    for _ in range(length):
        chars.append(_B32_ALPHA[n & 0x1F])
        n >>= 5
    return ''.join(reversed(chars))


def _b32_decode(s: str) -> int:
    """Decode a Base32 string to an integer (big-endian)."""
    n = 0
    for c in s:
        n = (n << 5) | _B32_IDX[c]
    return n


def _check_char(payload7: str) -> str:
    """
    Compute 1 Base32 check character from a 7-char Base32 payload.

    Uses a weighted sum with prime-ish weights that are all odd (coprime with 32),
    guaranteeing that every single-character substitution error is detected.
    Validated: 100% detection across all 248 possible single-char mutations.
    """
    val = sum(w * _B32_IDX[c] for w, c in zip(_CHECK_WEIGHTS, payload7)) % 32
    return _B32_ALPHA[val]


def split_seed_for_dm(seed: int) -> tuple[str, str]:
    """
    Encode a 32-bit seed into two 4-character Base32 shares for DataMatrix.

    Encoding pipeline:
        seed  →  Feistel-4 encrypt  →  32-bit ciphertext
              →  7 Base32 chars (35 bits, lower 32 used)
              →  + 1 weighted check char
              →  8-char payload  →  split at position 4

    Example (seed=920789066):
        ciphertext   = Feistel-4(920789066)  =  some 32-bit value
        payload7     = 'AM3ISC5'  (7 Base32 chars)
        check        = 'Y'
        full8        = 'AM3ISC5Y'
        share_a      = 'AM3I'
        share_b      = 'SC5Y'

    Both shares are encoded into 8x18 Data Matrix symbols:
        18 columns  x  0.417 mm/column  =  4.9 printer dots at 300 DPI
        (vs 16x48 old:  48 cols  x 0.156 mm  =  1.8 dots — fails on thermal)

    Returns (share_a: str, share_b: str), each exactly 4 Base32 characters.
    """
    ct       = feistel_encrypt(seed)
    payload7 = _b32_encode(ct, 7)
    check    = _check_char(payload7)
    full8    = payload7 + check          # e.g. 'AM3ISC5Y'
    return full8[:4], full8[4:]          # ('AM3I', 'SC5Y')


def recombine_seed_from_dm(share_a: str, share_b: str) -> int | None:
    """
    Recover the original seed from two 4-character Base32 shares.

    Decoding pipeline:
        share_a + share_b  →  8-char payload
                           →  verify check character  (returns None on failure)
                           →  Base32 decode 7 chars  →  32-bit ciphertext
                           →  Feistel-4 decrypt  →  original seed

    Returns seed (int) on success, or None if the check character fails
    (indicating a decode error, wrong pairing, or corrupted DM).
    """
    if not share_a or not share_b:
        return None
    # Accepts both trimmed 4-char strings and any surrounding whitespace
    full8 = share_a.strip() + share_b.strip()
    if len(full8) != 8:
        return None
    # Validate every character is in the Base32 alphabet
    if not all(c in _B32_IDX for c in full8):
        return None
    payload7, check_got = full8[:7], full8[7]
    if _check_char(payload7) != check_got:
        return None                       # integrity check failed
    ct   = _b32_decode(payload7) & 0xFFFFFFFF
    return feistel_decrypt(ct)

