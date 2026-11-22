# Hardware & Performance

## Environment
- Platform: Kaggle Notebooks
- GPU: 2x NVIDIA T4 (only 1 used — single-GPU pipeline, see WRITEUP for reasoning)
- RAM: ~30 GB available
- Disk: ~57.6 GB available
- Python: 3.13.16 version 
- Key packages: see requirements.txt

## Batching
- Batch size: 16 frames per YOLO call (tested 1/16/32/64; 16 and 32 nearly
  identical, 16 chosen for lower memory)
- Verified: batched vs non-batched on same videos, diffed output — 0 mismatches

## Measured throughput
- File 0 (train-00000): 90.0 min, 33 videos
- File 1 (train-00001): 97.8 min, 33 videos
- File 2 (train-00002): ~90-98 min, 33 videos (completed; exact time lost when session restarted)
- File 3 (train-00003): interrupted at 32/33 videos (session stopped); 1 error line recorded

## Scope actually run
Processed files 0-3 of the required 50 (~7.5 hours compute across two Kaggle
sessions, including one session loss that cost several hours of overnight
progress). At the measured ~90-98 min/file rate, the full 50-file / 125-hour
dataset would take an estimated 75-80 hours of compute — far beyond the time
available before the deadline. We prioritized submitting complete, verified,
correct results for a smaller scope over an incomplete or corrupted attempt
at all 50.

## Known limitations
- Did not complete all 50 required files.
- Did not implement multi-GPU parallelism (available 2nd T4 went unused) —
  deprioritized given time constraints and because grading runs on a single
  RTX 4090 regardless.