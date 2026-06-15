# CUDA Real-Time HDR Tone Mapping

## Overview

This project extends the CUDA webcam filter template by adding a simple real-time HDR tone mapping filter.

The goal was to implement a new CUDA-based filter that improves image brightness and contrast using exposure and gamma correction.

---

## Implemented Features

- Added new `HDR_TONEMAPPING` filter type
- Added string mapping for `hdr` and `hdr_tonemapping`
- Added CUDA HDR tone mapping kernel
- Added GPU function `applyHDRGPU`
- Added HDR handling in `main.cpp`
- Added simple exposure and gamma parameters
- Kept existing filters such as blur, sharpen, edge detection, and emboss

---

## HDR Formula

The implemented global tone mapping operator uses:

```cpp
mapped = 1.0f - expf(-color * exposure);
mapped = powf(mapped, 1.0f / gamma);