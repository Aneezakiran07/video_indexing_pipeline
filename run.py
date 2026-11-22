import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import pandas as pd
from ultralytics import YOLO

# these are the only objects we need to detect
target_classes = {"person", "car", "dog", "laptop", "cell phone"}
batch_size = 16

# load the pretrained YOLO model
model = YOLO("yolo11l.pt")


def aggregate_bucket(records):
    # total number of frames in this second
    total_frames = len(records)

    # get all unique objects detected in these frames
    all_classes_seen = set().union(*(objs for objs, _ in records)) if records else set()

    # an object is included only if it appeared in at least half of the frames
    objects_present = [
        cls for cls in all_classes_seen
        if sum(1 for objs, _ in records if cls in objs) / total_frames >= 0.5
    ]

    # get the person count from each frame and sort them to calculate the median
    person_counts = sorted(p for _, p in records)
    n = len(person_counts)
    mid = n // 2

    # if there is an odd number of frames, take the middle value
    if n % 2 == 1:
        median_people = person_counts[mid]
    else:
        # if there is an even number, take the average of the two middle values and round down
        median_people = (person_counts[mid - 1] + person_counts[mid]) // 2

    return sorted(objects_present), median_people


def process_video(video_bytes, video_filename, video_idx, out_file):
    # each second will have its own bucket containing the detected objects and person count for each frame
    buckets = defaultdict(list)
    cap = None

    def process_batch(frame_batch, timestamp_batch):
        # pass the batch of frames to YOLO instead of processing each frame separately
        results = model(frame_batch, conf=0.4, verbose=False)

        for result, ts in zip(results, timestamp_batch):
            # get the names of all objects YOLO detected in this frame
            names_found = [model.names[int(c)] for c in result.boxes.cls]

            # keep only the objects we are interested in
            names_found = [n for n in names_found if n in target_classes]

            # use the timestamp to decide which second this frame belongs to
            bucket_idx = int(ts)

            # count the actual number of people detected in this frame
            person_count = sum(1 for c in result.boxes.cls if model.names[int(c)] == "person")

            # store unique detected objects and the person count for this frame
            buckets[bucket_idx].append((set(names_found), person_count))

    try:
        # save the video bytes as an mp4 file so OpenCV can read it
        with open("sample.mp4", "wb") as f:
            f.write(video_bytes)

        # open the video and check if it was loaded successfully
        cap = cv2.VideoCapture("sample.mp4")
        if not cap.isOpened():
            raise ValueError("Could not open video")

        frame_batch, timestamp_batch = [], []

        # read the video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # get the timestamp of the current frame in seconds
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # keep the frame and its timestamp together
            frame_batch.append(frame)
            timestamp_batch.append(timestamp_sec)

            # once the batch is full, send it to YOLO for detection
            if len(frame_batch) == batch_size:
                process_batch(frame_batch, timestamp_batch)
                frame_batch, timestamp_batch = [], []

        # process any remaining frames that did not make a complete batch
        if frame_batch:
            process_batch(frame_batch, timestamp_batch)

    except Exception as e:
        # save the error so we can write it to the output file
        error_message = str(e)

    finally:
        # release the video after processing
        if cap is not None:
            cap.release()

        # aggregate the frames of each second and write the results
        for bucket_idx, records in sorted(buckets.items()):
            objects_present, median_people = aggregate_bucket(records)
            out_file.write(json.dumps({
                "video": f"{video_filename}:{video_idx}",
                "second": bucket_idx,
                "objects": objects_present,
                "people": median_people
            }) + "\n")

        # if something went wrong, write the error but keep the results already processed
        if 'error_message' in locals():
            out_file.write(json.dumps({
                "video": f"{video_filename}:{video_idx}",
                "error": error_message
            }) + "\n")


def main():
    parser = argparse.ArgumentParser()

    # take the input folder and output file path from the command line
    parser.add_argument("--input", required=True, help="folder of .parquet files")
    parser.add_argument("--output", required=True, help="path to write results.jsonl")
    args = parser.parse_args()

    # get all parquet files from the input folder in sorted order
    parquet_files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))

    # open the output file and process each parquet file
    with open(args.output, "w") as out_file:
        for path in parquet_files:
            file_name = os.path.basename(path)
            try:
                # read the parquet file into a dataframe
                df_file = pd.read_parquet(path)
            except Exception as e:
                # if a parquet file cannot be read, write an error and move to the next one
                out_file.write(json.dumps({
                    "video": f"{file_name}:0",
                    "error": f"could not read parquet file: {e}"
                }) + "\n")
                continue

            # each row contains one video, so process every row in the dataframe
            print(f"processing {file_name}: {len(df_file)} videos")
            for video_idx in range(len(df_file)):
                process_video(df_file.iloc[video_idx]["mp4"], file_name, video_idx, out_file)

    print("done")


if __name__ == "__main__":
    main()