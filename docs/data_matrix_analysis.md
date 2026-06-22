# Data Matrix Print Robustness Analysis

This document analyzes the relationship between payload size, Data Matrix symbol dimensions, and thermal-transfer print survivability at 300 DPI.

## 1. Current Reality

* **Is the Data Matrix always generated as 16x48?** 
  Yes. The `generate_cropped_dm` function is hardcoded to `size="16x48"` to prevent layout dimensions from randomly jumping when the `RectAuto` algorithm encounters varying data entropy.
* **Does payload length affect symbol complexity?** 
  Currently, **no**. Whether you provide 4 characters or 40 characters, the encoder generates a 16x48 module grid. The empty space is filled with padding codewords, creating unnecessary black/white modules.
* **Are we wasting capacity?** 
  **Massively.** An 8x32 symbol holds 16 ASCII characters. We are forcing a 16x48 symbol (which holds 49 ASCII characters) just to store an 8-character string, wasting roughly 83% of the symbol's capacity and unnecessarily shrinking the physical printed modules.

## 2. Smallest Possible Symbol

Using standard ASCII encoding, here are the minimum symbol sizes required for the proposed payload lengths:

| Encoding Scheme | Payload Length | Smallest Rectangular Symbol | Smallest Square Symbol |
| :--- | :--- | :--- | :--- |
| **A. Current (Hex + MD5)** | 8 chars | **8x32** (256 modules) | **14x14** (196 modules) |
| **B. XOR (Hex)** | 4 chars | **8x18** (144 modules) | **12x12** (144 modules) |
| **C. Feistel (Hex)** | 4 chars | **8x18** (144 modules) | **12x12** (144 modules) |
| **D. Feistel + Base32** | 3.5 chars | **8x18** (144 modules) | **12x12** (144 modules) |
| **E. Feistel + Checksum** | 5 chars | **8x18** (144 modules) | **12x12** (144 modules) |

> [!NOTE]
> An `8x18` rectangular symbol has a hard maximum capacity of **5 ASCII bytes**.

## 3. Printed Module Size Impact

**Constraint:** The Top Data Matrix spans the exact width of the CDP pattern.
**Given:** Pattern size = 7.5 mm. 
**Printer:** 300 DPI (1 dot ≈ 0.0846 mm).

Because the physical width is fixed to 7.5 mm, reducing the *number of columns* in the Data Matrix directly increases the physical size of each printed module.

| Symbol Size | Columns | Physical Module Width | Printer Dots (@ 300 DPI) | Print Status |
| :--- | :--- | :--- | :--- | :--- |
| **16x48** (Current) | 48 cols | 0.156 mm | **1.84 dots** | ❌ Fails (Muddy) |
| **8x32** (Current 8-char min) | 32 cols | 0.234 mm | **2.76 dots** | ⚠️ High Risk |
| **8x18** (New 4/5-char min) | 18 cols | 0.416 mm | **4.92 dots** | ✅ Safe |

**Impact:** Moving from a `16x48` symbol to an `8x18` symbol **increases physical module size by 166%**.

## 4. Thermal Printing Analysis

What contributes most to reliability?
1. **Larger module sizes (Most Critical):** Dot gain (ink spread) on a thermal printer is a physical constant—usually spreading about 0.5 to 1 dot per edge. If your module is only 1.8 dots wide (current reality), dot gain completely fills in the white space. If your module is 4.9 dots wide (`8x18`), dot gain only eats 1 dot, leaving 3.9 dots of clean white space.
2. **Smaller Data Matrix symbols:** A smaller symbol (fewer rows/cols) allows the layout engine to allocate more physical space per module. 
3. **Shorter payloads:** Only helpful because they *enable* the use of a smaller Data Matrix symbol.

## 5. Best Architecture Recommendation

**Winner: Feistel + 1-character Hex Checksum (5 chars per DM)**

* **DM A:** `[4-char hex ciphertext A]` + `[1-char hex CRC of ciphertext B]`
* **DM B:** `[4-char hex ciphertext B]` + `[1-char hex CRC of ciphertext A]`

**Why this is the best approach:**
1. **Density & Module Count:** 5 characters exactly maxes out the `8x18` capacity. You get the absolute lowest column count (18 cols) possible.
2. **Print Survivability:** At 4.92 dots/module, it guarantees 300 DPI survivability even with heavy dot gain.
3. **Local Validation:** Including a 1-character checksum allows the scanner to immediately cross-validate and pair the two shares before doing any cryptography or database lookups.
4. **Security/Obfuscation:** A Format-Preserving Encryption (FPE) Feistel cipher maps the 32-bit seed to a pseudorandom 32-bit ciphertext. This prevents sequential guessing and completely hides the seed.
5. **Why not Base32?** Hex is natively ASCII. Base32 offers no module-size reduction because both 4 hex chars and 3 Base32 chars fall under the 5-byte maximum of the `8x18` symbol. Using Hex avoids complex C40/ASCII switching logic in the barcode generator.

## 6. Code-Level Recommendation

**Recommendation: C. Use a fixed but smaller symbol (`size="8x18"`).**

You should **never** use auto-select (`RectAuto`) for a printed label. 
If `RectAuto` is used, a low-entropy seed might generate an `8x18` symbol, while a high-entropy seed generates an `8x32` symbol. Because the layout algorithm uses the module counts to calculate physical label dimensions, the total height of your printed label would randomly jump between two different sizes based on the specific seed being generated.

By adopting the 5-char payload and explicitly hardcoding `size="8x18"` in `generate_cropped_dm`, you guarantee that every label prints at exactly the same physical dimensions, with modules mathematically sized for 300 DPI thermal transfer success.

### Minimum Safe Pattern Widths

Assuming you want at least **4 printer dots** per module to survive dot gain:

| DPI | For `8x18` DM (18 cols) | For `8x32` DM (32 cols) | For `16x48` DM (48 cols) |
| :--- | :--- | :--- | :--- |
| **300 DPI** | **6.1 mm** | 10.8 mm | 16.3 mm |
| **600 DPI** | **3.0 mm** | 5.4 mm | 8.1 mm |

Since your pattern width is 7.5 mm, **`8x18` is the only symbol size capable of reliable 300 DPI printing.**
