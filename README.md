# 🔱 AEGIS-PROVENANCE
### *Universal Reality Auditor — Multimodal Forensic Engine for AI-Generated Video Detection*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <i>Five forensic layers. One probabilistic verdict. No hallucinations.</i>
</p>

---

## 📌 Table of Contents

1. [What is AEGIS-PROVENANCE?](#-what-is-aegis-provenance)
2. [Why This Matters](#-why-this-matters)
3. [System Architecture](#-system-architecture)
4. [The Five Forensic Layers — Deep Dive](#-the-five-forensic-layers--deep-dive)
   - [Layer 1: rPPG Biological Boundary Analysis](#layer-1--rppg-biological-boundary-analysis)
   - [Layer 2: Hardware PRNU Noise Fingerprinting](#layer-2--hardware-prnu-noise-fingerprinting)
   - [Layer 3: Spatial Entropy Analysis](#layer-3--spatial-entropy-analysis)
   - [Layer 4: Spectral Decay Ratio (Generator Fingerprint)](#layer-4--spectral-decay-ratio-generator-fingerprint)
   - [Layer 5: Metadata & Provenance Veto](#layer-5--metadata--provenance-veto)
5. [Dempster-Shafer Evidence Fusion](#-dempster-shafer-evidence-fusion)
6. [Tech Stack](#-tech-stack)
7. [Installation & Setup](#-installation--setup)
8. [Running the App](#-running-the-app)
9. [Interpreting Results](#-interpreting-results)
10. [Limitations & Known Edge Cases](#-limitations--known-edge-cases)
11. [Future Roadmap](#-future-roadmap)
12. [License](#-license)

---

## 🧠 What is AEGIS-PROVENANCE?

**AEGIS-PROVENANCE** is a research-grade, multimodal forensic engine designed to distinguish **real, camera-captured video** from **AI-generated or synthetically manipulated video** — without relying on any machine learning model that requires training data.

It operates entirely through **physics, information theory, signal processing, and probabilistic reasoning**, making it model-agnostic, bias-resistant, and generalizable across video sources.

Think of it as a **lie detector for video** — running five simultaneous scientific tests, each targeting a different physical or computational property that synthetic generation struggles to perfectly replicate. The outputs are then fused using **Dempster-Shafer Evidence Theory** to produce a single, mathematically grounded probability of authenticity.

> Built for an era where Sora, Runway, Kling, and HeyGen have made synthetic video indistinguishable to the human eye — but not yet to physics.

---

## 🌍 Why This Matters

| Problem | Scale |
|---|---|
| AI-generated deepfake videos in circulation | Growing exponentially |
| Detection tools that require labelled training data | Brittle, model-specific |
| Legal/journalistic need for forensic-grade evidence | High and underserved |
| Methods robust to compression, re-encoding, platform upload | Almost nonexistent |

AEGIS-PROVENANCE tackles the *hardest variant* of the problem: **compressed, re-uploaded video with no metadata trail**, where traditional methods fail.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     INPUT: Video File (.mp4 / .mov)                  │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
          ┌───────────────────▼────────────────────┐
          │        Frame-by-Frame Extraction        │
          │  (OpenCV VideoCapture, FPS-normalized)  │
          └──┬──────────┬──────────┬────────────────┘
             │          │          │
    ┌────────▼──┐  ┌────▼────┐  ┌─▼──────────┐
    │  Face ROI │  │  Full   │  │  Raw Video │
    │Extraction │  │  Frame  │  │   Header   │
    │(Haarcasc.)│  │Analysis │  │  & Footer  │
    └────┬──────┘  └────┬────┘  └─────┬──────┘
         │              │             │
  ┌──────▼──────┐ ┌─────▼──────────────────────────────┐
  │   LAYER 1   │ │         LAYERS 2 / 3 / 4            │
  │    rPPG     │ │  PRNU Noise | Entropy | Spectral    │
  └──────┬──────┘ └──────────────┬──────────────────────┘
         │                       │
         │         ┌─────────────▼──────┐
         │         │      LAYER 5       │
         │         │ Metadata/Provenance │
         │         └─────────────┬──────┘
         │                       │
  ┌──────▼───────────────────────▼──────┐
  │    DEMPSTER-SHAFER FUSION ENGINE    │
  │   (Probabilistic Evidence Combiner) │
  └──────────────────┬──────────────────┘
                     │
         ┌───────────▼────────────────┐
         │    FINAL FORENSIC VERDICT  │
         │  % Probability of Authenticity │
         └────────────────────────────┘
```

---

## 🔬 The Five Forensic Layers — Deep Dive

Each layer is an **independent evidence generator**. No layer is blindly trusted. Each produces a **confidence value between 0.0 and 1.0**, where values closer to 1.0 suggest authenticity and values closer to 0.0 suggest synthetic generation.

Before diving into the layers, it helps to understand one formula used across multiple layers: the **Asymmetric Sigmoid Map**. This is explained first so the layer descriptions can reference it cleanly.

---

### 📐 The Asymmetric Sigmoid Map (Shared Utility — Read First)

A **sigmoid** is an S-shaped mathematical curve that maps any real number to a value between 0 and 1. It is used throughout the engine to convert raw forensic measurements (like a noise ratio or entropy value) into smooth, bounded probability-like scores.

The standard sigmoid formula is:
```
σ(x) = 1 / (1 + e^(-x))
```

Where:
- `x` is the input value
- `e` is Euler's number, approximately 2.718 (the base of the natural logarithm)
- `e^(-x)` means e raised to the power of negative x
- As `x` increases toward +∞, the output smoothly approaches 1
- As `x` decreases toward -∞, the output smoothly approaches 0
- At `x = 0`, the output is exactly 0.5

AEGIS-PROVENANCE uses a **parameterized, asymmetric version** of this:

```
P(x) = 1 / (1 + e^( -scale × direction × (x - center) ))

Confidence = min_val + P(x) × (max_val - min_val)
```

**Every symbol defined:**

| Symbol | Type | Meaning |
|---|---|---|
| `x` | input | The raw forensic measurement (e.g., noise ratio, entropy value, spectral ratio) |
| `center` | parameter | The **inflection point** — the value of `x` where `P(x)` equals exactly 0.5. This is the decision boundary between suspicious and clean |
| `scale` | parameter | Controls the **steepness** of the S-curve. A high scale (e.g., 2000) makes the transition very sharp — a tiny change in `x` near `center` causes a large jump in output. A low scale (e.g., 4) produces a gentle, gradual curve |
| `direction` | parameter | Either `+1` or `-1`. `+1` = higher `x` → higher confidence (normal orientation). `-1` = higher `x` → lower confidence (inverted — used when a high measurement is suspicious) |
| `e^(...)` | math | Euler's number raised to the power of the expression in parentheses |
| `(x - center)` | math | How far the measurement is from the decision boundary — positive means above center, negative means below |
| `P(x)` | intermediate | A value strictly between 0 and 1 — the raw sigmoid output before scaling to the final range |
| `min_val` | parameter | The **floor** of the output range. Set to **0.10** across all layers — even the most suspicious reading retains 10% confidence, encoding epistemic humility (we are never 100% certain) |
| `max_val` | parameter | The **ceiling** of the output range. Set to **0.75–0.85** — even the cleanest reading can't claim full certainty |
| `Confidence` | output | Final confidence score, guaranteed to lie in `[min_val, max_val]` |

The asymmetry is deliberate: **the system fails aggressively (floor at 0.10) but passes cautiously (ceiling at 0.85)**. In forensics, falsely clearing a synthetic video is more dangerous than falsely flagging a real one.

---

### Layer 1 — rPPG Biological Boundary Analysis

**What it targets:** *The heartbeat hidden in your skin.*

#### Scientific Basis

**Remote Photoplethysmography (rPPG)** is the science of measuring a person's cardiac pulse from video alone, with no physical contact. Here is the underlying physics:

When the heart beats, it pumps blood in pulses through the body. Blood volume in **facial capillaries** (tiny blood vessels near the skin surface) rises and falls rhythmically with each heartbeat. Oxygenated blood absorbs more red and blue light and reflects more **green light**. This means the green channel of a person's facial skin oscillates very slightly at the heart rate — typically ~0.5–2% variation in pixel brightness, completely invisible to the human eye but measurable in video frames.

The normal human resting heart rate is **45–240 beats per minute (BPM)**. Converting to Hz (cycles per second): 1 Hz = 60 BPM. So the physiological pulse band is:
```
0.75 Hz  (= 45 BPM)  to  4.0 Hz  (= 240 BPM)
```

**Why AI video fails this test:**

AI video generators render faces as textures — they have no internal model of human circulatory physiology. Their green channel signal is either:
- **Too flat** — no oscillation at all (suspicious absence of any pulse)
- **Too perfect** — unrealistically high signal-to-noise ratio, because real human pulse signals always carry noise from breathing, micro-motion, and lighting variance

#### How It Works — Step by Step

**Step 1 — Face and Forehead Detection**

OpenCV's **Haar Cascade classifier** scans each frame for a frontal face. A Haar Cascade is a pre-trained detection model that uses simple rectangular features (differences between the average brightness of adjacent rectangular regions) to rapidly identify face regions.

Once a face bounding box is found with coordinates `(x, y, w, h)`:
- `x` = horizontal pixel coordinate of the top-left corner of the box
- `y` = vertical pixel coordinate of the top-left corner
- `w` = width of the bounding box in pixels
- `h` = height of the bounding box in pixels

The **forehead region** is extracted as a sub-region of the face box:
```
x_min = x + 0.30 × w     ← 30% from left edge
x_max = x + 0.70 × w     ← 70% from left edge
y_min = y + 0.05 × h     ← 5% from top edge
y_max = y + 0.20 × h     ← 20% from top edge
```

The forehead is chosen because it has minimal muscle movement, no hair occlusion, and sits directly above major facial arteries — giving the highest-quality rPPG signal.

**Motion Stabilization via Exponential Moving Average (EMA):**

If the face box jumps between frames due to head movement, an EMA smooths the position:
```
x_stable = 0.90 × x_previous + 0.10 × x_current
```
- `0.90` = weight given to the previous (stable) position
- `0.10` = weight given to the new (potentially jittery) detection
- 90% of the position is inherited from the last frame; only 10% updates each step

**Motion Abstention:** If face displacement between frames exceeds 15% of the face's own width (indicating heavy head movement), the layer returns a neutral `0.50` confidence and skips analysis for that video. Motion artifacts completely contaminate the rPPG signal; an abstention is more honest than a corrupted verdict.

**Step 2 — Green Channel Signal Extraction**

For each frame, the **mean green channel value** of the forehead patch is computed and appended to a growing time-series array. Over a 30-second video at 30 FPS, this builds a 900-point signal.

**Step 3 — Detrending**

The raw green signal has slow, non-oscillatory drifts caused by changing room lighting or camera auto-exposure adjustments. `scipy.signal.detrend` removes this by fitting a **least-squares linear trend** to the signal and subtracting it. After detrending, only the oscillatory components (including the pulse) remain.

**Step 4 — Butterworth Bandpass Filter**

A **Butterworth filter** is a signal processing filter that passes frequencies within a specified band and suppresses (attenuates) everything outside it. "Butterworth" refers to the fact that the passband is maximally flat — no ripples in the frequencies you want to keep.

```
Passband: 0.75 Hz to 4.0 Hz
```

**What "5th-order" means:** Filter order controls how sharply frequencies outside the band are cut. Each order of a Butterworth filter adds 20 dB/decade of rolloff slope. A 5th-order filter suppresses out-of-band frequencies at 100 dB/decade — meaning signals just outside the 0.75–4.0 Hz range are very aggressively attenuated. Higher order = sharper cutoff, but also more computational phase distortion.

`scipy.signal.filtfilt` applies the filter **twice** — once forward through the signal, once backward. This **double-pass** cancels all phase distortion, producing a zero-phase filtered output (no time-shifting of the signal).

**Step 5 — Welch's Power Spectral Density Estimation**

**Power Spectral Density (PSD)** answers: *"How much signal power (energy per unit time) exists at each frequency?"*

**Welch's method** estimates the PSD robustly by:
1. Dividing the signal into overlapping segments
2. Computing the **Fast Fourier Transform (FFT)** of each segment — converting from time-domain (amplitude vs. time) to frequency-domain (power vs. frequency)
3. Averaging the FFT magnitude-squared values across all segments

Averaging reduces random estimation noise. The output is two arrays:
- `f` — array of frequency values in Hz
- `pxx` — array of power values at each frequency in `f`

**Step 6 — SNR Computation**

```
SNR = Σ pxx[pulse_band] / (Σ pxx[all] + ε)
```

**Every symbol defined:**

| Symbol | Meaning |
|---|---|
| `SNR` | Signal-to-Noise Ratio — the fraction of total signal power that falls within the physiological heartbeat frequency range (0.75–4.0 Hz) |
| `Σ` | Summation operator — add up all values in the array that satisfy the condition in brackets |
| `pxx[pulse_band]` | The subset of power values at frequencies between 0.75 Hz and 4.0 Hz |
| `pxx[all]` | Power values at all frequencies in the spectrum |
| `ε` (epsilon) | A tiny constant (~1×10⁻⁸) added to the denominator to prevent division by zero when the signal is completely flat |
| Result range | 0.0 (no power in pulse band) to 1.0 (all power in pulse band) |

**Step 7 — Verdict Assignment**

```
SNR > 0.60  →  confidence = 0.10   (impossibly perfect signal → almost certainly AI)
SNR < 0.015 →  confidence = 0.40   (no detectable pulse → likely fake or destroyed by compression)
Otherwise   →  confidence = 0.75   (natural pulse variance → consistent with a real human)
```

---

### Layer 2 — Hardware PRNU Noise Fingerprinting

**What it targets:** *The camera sensor's unique microscopic fingerprint.*

#### Scientific Basis

Every digital camera sensor is a grid of **photodiodes** — tiny semiconductor elements that convert incoming photons (light particles) into electrical charge. Due to microscopic imperfections in the silicon manufacturing process (called **doping variations** — controlled introduction of impurities into silicon to adjust its electrical properties), each photodiode responds slightly differently to the same amount of light.

This creates a fixed, deterministic, per-pixel sensitivity pattern called **Photo Response Non-Uniformity (PRNU)** — effectively a unique fingerprint for every physical camera sensor. It is invisible to the eye but statistically present in every image taken by that camera.

**AI-generated frames have no camera** — they were produced by a GPU running a neural network. Therefore they have no PRNU fingerprint. Whatever noise they contain comes from the generator's learned approximations, which have statistically different properties from real sensor noise.

#### How It Works — Step by Step

**Step 1 — Noise Residual Extraction**

Each frame is resized to 512×512 and converted to grayscale to reduce processing cost.

OpenCV's `fastNlMeansDenoising` removes random (Gaussian) noise using a technique called **Non-Local Means denoising**: for each pixel, it searches for similar-looking patches elsewhere in the image and averages them together — suppressing random per-pixel noise while preserving repeating structured content (like the PRNU pattern).

The noise residual (what was removed) is:
```
noise_residual = |original_frame − denoised_frame|
```
- `|...|` denotes absolute value — the pixel-by-pixel difference, always positive
- What remains is the fixed-pattern component that survived denoising — ideally, the PRNU fingerprint

**Step 2 — NSR Computation**

```
NSR = Var(noise_residual) / (Var(original_image) + ε)
```

**Every symbol defined:**

| Symbol | Meaning |
|---|---|
| `NSR` | Noise-to-Signal Ratio — what fraction of the image's total variance is explained by the extracted noise residual |
| `Var(x)` | **Variance** of the pixel array `x`. Variance measures how spread out the pixel values are around their mean: `Var = (1/N) × Σᵢ (xᵢ − x̄)²`, where `x̄` is the mean pixel value, `xᵢ` is each individual pixel value, `N` is the total number of pixels, and `Σᵢ` means sum over all pixels |
| `noise_residual` | The absolute difference between original and denoised frame |
| `original_image` | The grayscale frame before denoising |
| `ε` (epsilon) | Small constant (~1×10⁻⁶) to prevent division by zero |

**Step 3 — Confidence Mapping**

NSR is passed through the asymmetric sigmoid with:
- `center = 0.0008` — empirically calibrated so that moderately compressed real videos safely register above this threshold
- `scale = 2000` — very steep curve; a tiny deviation in NSR near the center causes a large confidence change
- `direction = +1` — higher NSR means more noise, which is more consistent with real camera hardware
- `max_val = 0.75`

---

### Layer 3 — Spatial Entropy Analysis

**What it targets:** *The information complexity of a video frame.*

#### Scientific Basis

**Shannon Entropy**, introduced by Claude Shannon in 1948, quantifies the average unpredictability (or information content) of a signal. The core intuition: if you know a signal perfectly in advance (e.g., all pixels are the same color), its entropy is zero — it carries no new information. If every pixel is completely unpredictable, entropy is maximized.

Applied to images: real-world video frames contain rich, complex textures from environmental stochasticity — lighting variations, material surfaces, atmospheric effects, and sensor noise — all contributing high entropy. AI-generated frames often have:
- **Too-low entropy** — overly smooth, plastic-looking, uniform textures
- **Too-high entropy** — aggressive sharpening artifacts creating artificially complex noise patterns

#### How It Works — Step by Step

**Step 1 — Histogram Construction**

The frame is resized to 512×512 and converted to grayscale. Each pixel has an integer intensity value from 0 (black) to 255 (white).

A **256-bin histogram** counts how many pixels in the frame have each intensity level:
- Bin `i` = number of pixels with intensity value `i`, for `i` from 0 to 255

The histogram is then **normalized**: every bin count is divided by the total number of pixels (`512 × 512 = 262,144`). This converts raw counts into probabilities:
- `p(i)` = proportion of pixels with intensity `i`
- All `p(i)` values sum to exactly 1.0

**Step 2 — Shannon Entropy Calculation**

```
H = −Σᵢ p(i) × log₂(p(i) + ε)
```

**Every symbol defined:**

| Symbol | Meaning |
|---|---|
| `H` | Shannon Entropy — the final entropy value, measured in **bits** |
| `−` | Negative sign. Since `p(i)` is between 0 and 1, `log₂(p(i))` is always ≤ 0 (log of a fraction is negative). The minus sign flips the whole expression positive |
| `Σᵢ` | Summation over all `i` from 0 to 255 — compute the product `p(i) × log₂(p(i))` for every intensity bin and add all 256 results together |
| `p(i)` | Normalized probability of intensity value `i` — what fraction of the frame's pixels have this brightness |
| `log₂(...)` | Logarithm base 2. This is used because entropy is measured in **bits** (binary units). `log₂(0.5) = −1` (50/50 probability contributes exactly 1 bit of uncertainty). `log₂(1.0) = 0` (certainty contributes zero bits) |
| `ε` (epsilon) | Small constant added inside the log to prevent `log₂(0)`, which is mathematically undefined (negative infinity) |
| Unit: **bits** | Maximum theoretical entropy for a 256-level image = `log₂(256) = 8 bits`, achieved when all 256 intensity levels are equally probable |

**Step 3 — Confidence Mapping**

Passed through the asymmetric sigmoid with `center = 7.0` bits and `scale = 4.0`. Values near 7.0 bits correspond to natural image complexity. Values significantly above or below map to lower confidence.

---

### Layer 4 — Spectral Decay Ratio (Generator Fingerprint)

**What it targets:** *The invisible frequency-domain artifacts that AI generators embed in their frames.*

#### Scientific Basis

**Spatial Frequency:** Every image can be described in two equivalent ways:
1. The **spatial domain** — pixel values at each (x, y) location (what you see)
2. The **frequency domain** — how rapidly pixel values oscillate across the image

Low spatial frequencies correspond to slow, gradual changes (large background regions, smooth gradients). High spatial frequencies correspond to rapid changes (sharp edges, fine textures, noise).

The **2D Fast Fourier Transform (FFT)** mathematically converts an image from the spatial domain to the frequency domain.

**The 1/f Natural Law:** Natural photographs follow a well-established statistical law: their frequency-domain power decays proportionally to `1/f` as frequency `f` increases. This means low-frequency content dominates, and energy decreases smoothly toward higher frequencies. This is a property of the physical world (natural scenes, textures, lighting) and real camera optics.

**Why AI generators violate this:** Generative models introduce characteristic high-frequency artifacts:
- **GAN checkerboard artifacts** — from transposed convolution ("deconvolution") operations in the generator network. The kernel steps create periodic grid-like patterns at specific spatial frequencies
- **Diffusion model ringing** — overshoot artifacts at edges caused by approximation errors in the score function
- **Over-sharpened textures** — caused by adversarial training pressure to produce "crisp-looking" but physically unnatural detail

These artifacts elevate the high-frequency energy beyond the natural `1/f` range.

#### How It Works — Step by Step

**Step 1 — 2D FFT**

```python
f_transform = np.fft.fft2(gray_frame)
```

`fft2` computes the **2D Discrete Fourier Transform** of the 512×512 grayscale image. It produces a 512×512 array of **complex numbers** — each cell `F(u, v)` represents the amplitude and phase of a specific 2D spatial frequency with horizontal frequency `u` and vertical frequency `v`.

**Step 2 — fftshift**

```python
f_shift = np.fft.fftshift(f_transform)
```

By default, numpy's FFT places the **DC component** (the zero-frequency term — representing the image's overall average brightness) at coordinate (0, 0) — the top-left corner. High frequencies appear near the center, which is counterintuitive to visualize.

`fftshift` **swaps the four quadrants** of the 512×512 frequency array so that:
- The DC component (zero frequency) moves to the **center** of the array at `(256, 256)`
- Low frequencies surround the center
- High frequencies appear at the **edges** of the array

After `fftshift`: center = low frequency, edges = high frequency. This makes spatial frequency directly proportional to radial distance from the center.

**Step 3 — Magnitude Spectrum**

```
magnitude_spectrum = 20 × log(|f_shift| + ε)
```

**Every symbol defined:**

| Symbol | Meaning |
|---|---|
| `f_shift` | The FFT output after quadrant rearrangement — a 512×512 array of complex numbers |
| `|f_shift|` | **Magnitude** of each complex FFT coefficient. A complex number has the form `a + bi`; its magnitude is `√(a² + b²)`. This extracts just the amplitude (how strong each frequency component is), discarding the phase |
| `log(...)` | Natural logarithm (base `e`). Applied to compress the enormous dynamic range of FFT magnitudes into a manageable scale |
| `20 ×` | Converts amplitude to **decibels (dB)** — a logarithmic unit for amplitude ratios. `20 × log₁₀(A)` in dB. Here `log` base `e` is used but the factor 20 follows the dB convention for expressing relative signal levels |
| `ε` (epsilon) | Small constant (~1×10⁻⁸) to prevent `log(0)`, which is undefined |
| Output | A 512×512 array of real-valued magnitudes in dB — the **magnitude spectrum** |

**Step 4 — Radial Frequency Masks**

The center of the shifted spectrum is at `(cy, cx) = (256, 256)`. For every pixel `(y, x)` in the spectrum, its **radial distance** from the center is:

```
r = √( (x − cx)² + (y − cy)² )
```

| Symbol | Meaning |
|---|---|
| `r` | Radial distance — how far this frequency cell is from the DC component (zero frequency). A larger `r` means a higher spatial frequency |
| `(x − cx)` | Horizontal offset from center |
| `(y − cy)` | Vertical offset from center |
| `√(...)` | Square root — gives the Euclidean (straight-line) distance |

Two circular frequency masks are created based on this radius:

```
Low-Frequency (LF) mask:   r < 50    ← inner circle, large structures and gradients
High-Frequency (HF) mask:  r > 150   ← outer ring, fine detail, noise, and artifacts
```

**Step 5 — HF/LF Ratio**

```
HF_LF_ratio = mean(magnitude_spectrum[HF_mask]) / mean(magnitude_spectrum[LF_mask])
```

| Symbol | Meaning |
|---|---|
| `mean(...)` | Arithmetic mean (average) of the magnitude values across all pixels inside the specified mask |
| `magnitude_spectrum[HF_mask]` | All magnitude values at radial distances > 150 (high-frequency ring) |
| `magnitude_spectrum[LF_mask]` | All magnitude values at radial distances < 50 (low-frequency center) |
| `HF_LF_ratio` | How much high-frequency energy exists relative to low-frequency energy. Natural images: low ratio (LF dominates). AI frames with artifacts: elevated ratio |

**Step 6 — Confidence Mapping**

Passed through the asymmetric sigmoid with `direction = −1` (inverted): a **higher** HF/LF ratio is **more suspicious**, so it maps to **lower** confidence. Decision boundary at `center = 0.62`.

---

### Layer 5 — Metadata & Provenance Veto

**What it targets:** *The paper trail that AI generators sometimes forget to sanitize.*

#### How It Works

Every video file is a binary container format (MP4, MOV) that wraps the compressed video stream with **metadata** — text fields embedded directly into the binary file describing how it was created, encoded, and processed. These live in the file's **header** (the opening bytes) and sometimes the **footer** (the closing bytes).

The engine opens the raw binary file and reads the first and last 8,192 bytes (8 KB each), decoding them as UTF-8 text (silently ignoring non-decodable bytes). This text is searched for known AI generator signature strings:

| Generator | Signature Tag |
|---|---|
| Runway ML | `Runway` |
| Midjourney Video | `Midjourney` |
| HeyGen | `HeyGen` |
| OpenAI Sora | `Sora` |
| Google Veo | `Veo` |
| Google SynthID | `SynthID` |
| Luma Dream Machine | `Luma` |
| Pika Labs | `Pika` |
| Kling AI | `Kling` |
| Krea AI | `Krea` |

**This layer operates as a HARD VETO.** If any signature is found:
- `meta_conf` is set to `0.01`
- After the Dempster-Shafer fusion, the final score is **overridden to `0.01`** — this single layer can condemn the video regardless of what all other layers concluded

Strings like `Lavf` (from FFmpeg's libavformat — used when legitimately re-encoding real video) and `ffmpeg` are intentionally excluded from the trigger list. This prevents false positives on real footage that was edited or compressed using standard tools.

If no AI signature is found, the layer returns a neutral `0.50` — it abstains. Absence of metadata is not proof of authenticity (many workflows strip metadata), so the layer does not reward "clean" files.

---

## ⚖️ Dempster-Shafer Evidence Fusion

**Why not just average the five confidence scores?**

Simple averaging treats all evidence as equally reliable and independent. In forensic analysis:
- Some layers abstain (return 0.50) when conditions aren't right — those shouldn't count the same as an active verdict
- Two agreeing sources should reinforce each other's conviction more than their average suggests
- Two disagreeing sources should express more uncertainty than their average suggests
- One hard veto should dominate regardless of what others say

**Dempster-Shafer Theory (DST)** is a mathematical framework for reasoning under uncertainty, generalizing Bayesian probability to handle evidence that is incomplete, ambiguous, or internally conflicting — exactly the forensic scenario.

#### Core Concepts

**The Two Hypotheses:**
```
H_REAL = "This video is authentic (camera-captured)"
H_FAKE = "This video is synthetic (AI-generated)"
```

**Basic Probability Assignment (BPA) / Mass Function:**

Each layer assigns a **mass** to each hypothesis — how much of its total belief it commits to each possibility:
```
m(H_REAL) = the layer's confidence score
m(H_FAKE) = 1 − the layer's confidence score
```

- `m(H_REAL) + m(H_FAKE) = 1.0` always — total belief is fully distributed
- Uncertainty is already encoded in the asymmetric map's conservative output range (0.10 to 0.85)

#### The Dempster Combination Rule

To combine two layers' mass functions `m₁` (Layer A) and `m₂` (Layer B):

**Step 1 — Compute the Conflict Factor K:**

```
K = m₁(REAL) × m₂(FAKE)  +  m₁(FAKE) × m₂(REAL)
```

**Every symbol defined:**

| Symbol | Meaning |
|---|---|
| `K` | The **conflict factor** — how much total probability mass is assigned to contradictory conclusions. Ranges from 0 (perfect agreement) to 1 (total contradiction) |
| `m₁(REAL) × m₂(FAKE)` | The probability mass where source 1 says REAL *and* source 2 says FAKE — direct conflict |
| `m₁(FAKE) × m₂(REAL)` | The probability mass where source 1 says FAKE *and* source 2 says REAL — direct conflict |
| `K` near 0 | Sources largely agree — combination amplifies their shared conclusion |
| `K` near 1 | Sources fundamentally disagree — combination produces high uncertainty |
| `1 − K` | The **normalizing denominator** — the total probability mass that is NOT in conflict. Dividing by this renormalizes the result so combined masses still sum to 1 |

When `K` approaches 1.0 (total contradiction), the denominator approaches zero and the formula becomes numerically unstable. The code caps `K` at 0.999 for safety.

**Step 2 — Compute the Combined Mass:**

```
m_combined(REAL) = [ m₁(REAL) × m₂(REAL) ] / (1 − K)

m_combined(FAKE) = [ m₁(FAKE) × m₂(FAKE) ] / (1 − K)
```

**Every symbol defined:**

| Symbol | Meaning |
|---|---|
| `m₁(REAL) × m₂(REAL)` | The mass where **both sources agree** on REAL — this is amplified after normalization |
| `m₁(FAKE) × m₂(FAKE)` | The mass where **both sources agree** on FAKE — this is amplified after normalization |
| `/ (1 − K)` | Division by the non-conflicting mass — renormalizes so `m_combined(REAL) + m_combined(FAKE) = 1.0` |

**Intuition in plain English:** When two independent sources agree, their combined conviction is *stronger* than either source alone. When they disagree, their combined belief is *weaker* — the conflict dilutes both sides.

#### Iterative Fusion Across All Five Layers

The five layers are combined sequentially — each step folds one more layer's evidence into the running combination:

```
Step 1:  m₁₂    = DS_combine( m_L1, m_L2 )
Step 2:  m₁₂₃   = DS_combine( m₁₂, m_L3 )
Step 3:  m₁₂₃₄  = DS_combine( m₁₂₃, m_L4 )
Step 4:  m₁₂₃₄₅ = DS_combine( m₁₂₃₄, m_L5 )

Final score = m₁₂₃₄₅(REAL)
```

**Post-Fusion Metadata Override:**

After all layers are fused, the metadata veto is applied:
```
if meta_conf ≤ 0.05:
    final_score = 0.01   ← hard override, no appeal
```

#### Why DST Beats Simple Averaging — A Concrete Example

Say Layer 1 gives `0.85` (strong REAL signal) and Layer 2 gives `0.15` (strong FAKE signal):

- **Simple average**: `(0.85 + 0.15) / 2 = 0.50` — completely uninformative, looks like uncertainty
- **Dempster-Shafer**:
  - `K = 0.85 × 0.85 + 0.15 × 0.15 = 0.745` ← high conflict
  - Combined REAL mass = `(0.85 × 0.15) / (1 − 0.745) ≈ 0.50`
  - Combined FAKE mass = `(0.15 × 0.85) / (1 − 0.745) ≈ 0.50`
  - The system correctly recognizes **genuine conflict** and expresses appropriate agnosticism — rather than falsely appearing balanced like the average, DST explicitly marks this as *unresolved evidence*

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Computer Vision | OpenCV |
| Signal Processing | SciPy (`butter`, `filtfilt`, `detrend`, `welch`) |
| Numerical Computing | NumPy |
| Visualization | Plotly |
| FFT Analysis | NumPy FFT |
| Face Detection | OpenCV Haar Cascades |

---

## 📦 Installation & Setup

### Prerequisites

- Python **3.9 or higher**
- `pip` (Python package manager)
- A terminal / command prompt

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Multi-Modal-Synthetic-Video-Verification-Engine.git
cd Multi-Modal-Synthetic-Video-Verification-Engine
```

---

### Step 2 — Create a Virtual Environment

A **virtual environment** is an isolated Python installation scoped to this project alone. It prevents dependency version conflicts with other Python projects on your system and keeps your global Python installation clean.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` prepended to your terminal prompt, confirming the environment is active.

---

### Step 3 — Install Dependencies

```bash
pip install streamlit opencv-python numpy scipy plotly
```

Or, if a `requirements.txt` is provided:
```bash
pip install -r requirements.txt
```

<details>
<summary>📋 Full dependency list (click to expand)</summary>

```
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
plotly>=5.17.0
```

</details>

---

### Step 4 — Verify Installation

```bash
python -c "import streamlit, cv2, numpy, scipy, plotly; print('All dependencies OK')"
```

You should see: `All dependencies OK`

---

## ▶️ Running the App

With your virtual environment **activated** and inside the project directory:

```bash
streamlit run app.py
```

Streamlit will automatically open the app in your default browser. If it doesn't open automatically, navigate to:

```
http://localhost:8501
```

**To stop the app:** Press `Ctrl + C` in the terminal.

---

### Using the Application

1. **Upload a video** using the file uploader (`.mp4` or `.mov` formats supported)
2. The engine begins **live frame-by-frame analysis** — you will see:
   - A live video feed with face and forehead ROI overlaid in green
   - A real-time rPPG pulse waveform chart
   - Layers 2 & 3 numerical metrics updating in real time
   - Layer 4 & 5 status panel
   - A progress bar tracking overall processing
3. Once processing completes, the **Final Forensic Report** appears below, showing:
   - Overall verdict (Authenticity Verified / Synthetic Media Detected)
   - Dempster-Shafer combined probability of authenticity
   - Individual confidence scores for all five layers with their raw measurements

---

## 📊 Interpreting Results

### Final Probability Score

| Score | Interpretation |
|---|---|
| **> 75%** | High confidence — likely authentic, camera-captured video |
| **50–75%** | Uncertain — inconclusive evidence, or heavily compressed footage |
| **25–50%** | Likely synthetic — multiple forensic markers present |
| **< 25%** | High confidence — strong evidence of AI generation or manipulation |
| **~1%** | Hard veto triggered — AI generator metadata signature detected |

### Per-Layer Confidence Guide

| Layer | Conf ~0.75–0.85 | Conf ~0.40–0.50 | Conf ~0.10–0.20 |
|---|---|---|---|
| L1 rPPG | Natural pulse variance detected | Abstained (high motion or no face) | Implausibly perfect or absent pulse |
| L2 PRNU | Sensor noise consistent with real camera | Borderline NSR | NSR inconsistent with hardware capture |
| L3 Entropy | Rich, natural image complexity | Moderate information content | Over-smooth or unnaturally sharp textures |
| L4 Spectral | Natural HF/LF energy distribution | Borderline spectral ratio | Generator frequency artifacts detected |
| L5 Metadata | — | Clean / stripped metadata | AI generator signature found (hard veto) |

---

## ⚠️ Limitations & Known Edge Cases

- **Heavy recompression**: Uploading real video to social platforms and re-downloading can degrade PRNU signals (Layer 2) beyond recovery. The engine is calibrated to tolerate moderate compression but may abstain on extreme cases.
- **No face in frame**: If no face is detected throughout the video, Layer 1 abstains entirely, reducing overall discriminative power.
- **Unknown AI generators**: Layer 5 only vetoes generators with known metadata signatures. New tools or deliberate metadata sanitization bypass this layer.
- **Very short clips**: Layer 1 requires at least 3 seconds of footage to compute a meaningful rPPG SNR estimate.
- **Screen recordings of real video**: May confuse Layers 2 and 4 due to monitor-induced noise patterns being superimposed on the real footage.

---

## 🚀 Future Roadmap

- [ ] Temporal consistency analysis across frames (GAN flicker detection)
- [ ] Audio-visual synchronization coherence check
- [ ] Expanded and continuously updated metadata signature database
- [ ] Batch processing mode for multiple video analysis
- [ ] Exportable PDF forensic report
- [ ] REST API wrapper for integration into media verification pipelines
- [ ] Benchmarking against FaceForensics++, DFDC, and GenVideo-Bench

---

## 👤 Author

**Soumik Sinha**  
B.Tech CSE, PES University, Bengaluru  
[GitHub](https://github.com/SoumikSinha) · [LinkedIn](http://linkedin.com/in/soumik-sinha-928a21352)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built to stay one step ahead of synthetic reality.</i><br>
  <b>🔱</b>
</p>
