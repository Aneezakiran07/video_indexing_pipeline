# Writeup
## 1 timings 

I completed only 4 of the 50 required files due to time constraints on my side. Files 0, 1 and 2 are completed and each file had 33 videos. File 3 stopped at 32 video and that error also got recorded in our json. So, i processed 131 of about 1,650 videos.
It took 90 min for file 0 and 97.8 min for file 1. I did rough estimate (about 94 min per file x 50 files) and it would have cost about 78 hours for 50 files. My first intuition was to sent one frame per YOLO call, and for it , it took 5.54s for 3 videos (first 60 frames of each). Then i did batching and sent 16 frames in one GPU call, which took 4.68s for the same videos, so about 1.18x faster. Scaling it up, one frame at a time would be about 93 hrs, so batching would have saved us approx 14 hrs. These are only estimates, i never ran the full 50 files.
## 2 pipeline
First, the pipeline reads each parquet file in input folder. And for every row, it writes mp4 bytes to a temp file. Then it opens it with openCv. It reads frame one at a time, records its timestamp, and collects 16  frames into a batch.
Each batch has 0.4 confidence threshold in YOLO. Then pipeline filters the required 5 objects. Then bucketing step came, in which each frame detection goes into a bucket based on its timestamp. Its like first bucket contains only frames between 0 to 0.999 second. so bucket 0 is for second one and so on.
Then, a aggregated step  is performed in which according to task's requirement, count object only if it appears at least 50%time. While the people value is median per frame person count. Finally, one JSON line/second is written. It has video id, the seond and objects and people count. Incase of any error, processing dont stop and error line is written in JSONL.

## 3 Where the time goes
Firstly, i had suspected that OPENCV decoding was having most of cost, but then i ran a script and find out Decoding is only 0.34s while inference is taking 27.03s. This was tested on one video with 929 frames, and i didnt time writing the video to disk or reading parquet separately. I also tried to do batching using 32 frames in one go, but it was not saving much time (4.91s vs 4.68s for 16) while 64 frames worsened it more (5.02s), so larger batched didnt help much.
I also thought about using second GPU and dividing independt work among it, but i didnt do it. Grading runs on a single RTX 4090 so second GPU wont help there, and running many video processes on one GPU also wont give speedup. YOLO already runs on CUDA too, so just "using CUDA" is not a new idea.

## 4 Checking correctness
After adding batching, I ran the one frame at a time version and the batched version on the same 3 videos (first 60 frames each). Then i compared every second objects and people count. There were 0 mismatches out of 6 s. To make sure bucketing by timestamp is a good choice, i also checked frame timestamp on three videos , and they increased steadily with constant step. (all these steps are documented greatly in my notebook)
Limits are the comparison test was small, I did not compare against a per-frame reference. and I didnt verify rotation tag handling

## 5 Working with AI
Ai said batching frames would be faster than one frame at a time,, and one frame per call is main bottleneck. I didnt just trust it, so i tested batch size 1, 16, 32 and 64 myself on 3 videos. Batch 16 took 4.68s vs 5.54s for batch 1, only about 1.18x faster, while using 32 and 64 made processing time increase instead. Then in the end i found out inference is 98.7% of the total runtime, so batching helped only a little and the model itself is the bottleneck.

## 6. Next steps
The biggest gap is my incomplete run, so i will run and finish all 50 files. Then I will work on rotated videos. Then i will try FP16 inference on the 4090 since inference is most of the time, and run the comparison again to make sure output didnt change. Moreover, i will add a resume support also so scripts continues from last completed video and session loss on kaggle notebook doesnt cost hrs of progress.