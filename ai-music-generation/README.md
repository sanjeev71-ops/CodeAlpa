# AI Music Generation with LSTM

An AI music generation project that learns melodic patterns from MIDI files and generates new melodies using a Long Short-Term Memory (LSTM) neural network.

## Project Overview

This project uses the Tegridy MIDI Dataset to train a sequence-based deep learning model for symbolic music generation.

The MIDI files are processed with `music21`, converted into representative melody events containing pitch and duration, encoded into integer tokens, and used to train an LSTM model with PyTorch.

The generated event sequence is then converted back into a MIDI file and can be rendered to WAV audio using FluidSynth and a General MIDI SoundFont.

## Dataset

The project uses the **Tegridy MIDI Dataset**.

For this project:

- MIDI files available in the selected dataset archive: **125**
- MIDI files successfully parsed: **125**
- Failed MIDI files: **0**
- Total MIDI elements analyzed: **89,576**
- Representative melody events: **68,904**
- Unique melody events: **2,068**

The dataset contains MIDI files with varying durations and musical structures.

### MIDI Statistics

| Statistic | Events | Duration |
|---|---:|---:|
| Mean | 716.61 | 190.04 |
| Minimum | 281 | 88.00 |
| Median | 669 | 180.00 |
| Maximum | 1,508 | 437.00 |

## Data Processing

The MIDI files are parsed using `music21`.

The processing pipeline is:

```text
MIDI files
    ↓
music21 MIDI parsing
    ↓
MIDI element analysis
    ↓
Representative pitch extraction
    ↓
Pitch + duration events
    ↓
Event vocabulary
    ↓
Integer encoding
    ↓
Fixed-length training sequences
