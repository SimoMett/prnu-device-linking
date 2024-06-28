import sys

import ffmpeg
import glob
import os
from params import *


# extract clip WITHOUT audio, but for the purpose of this project it's fine
def extract_clip(input_file: str, output: str, start=0, duration=10):
    (
        ffmpeg
        .input(input_file)
        .trim(start=start, duration=duration)
        .output(output)
        .run()
    )


# This function was a pain in the arse
# I'll try to doc it
def concatenate_videos(videos: list, output: str):
    default_start = 0
    default_duration = 10

    # For each video sources I have to trim the video and create a split filter using 'filter_multi_output'
    # and then count every stream occurrence used during encoding process to avoid using the same stream multiple times
    # (otherwise I'll get the error "ValueError: encountered trim with multiple outgoing edges...")
    videos_set = set(videos)  # no duplicates
    splits_map = dict.fromkeys(videos_set)
    for v in splits_map.keys():
        assert os.path.isfile(v)
        splits_map[v] = (
            [ffmpeg.input(v).trim(start=default_start, end=default_duration).filter_multi_output("split"), 0])

    # Now I build the clips sequence allocating the streams of each sources
    clips = []
    for v in videos:
        clips.append(splits_map[v][0].stream(splits_map[v][1]))
        splits_map[v][1] += 1

    # Finally, stretch every clip to 1080p and generate output
    partial_output = output.replace(".mp4", ".partial.mp4")
    (
        ffmpeg.concat(*[c.filter("scale", "1920-1080").filter("setsar", "1-1") for c in clips])
        .output(partial_output)
        .overwrite_output()
        .run()
    )
    os.rename(partial_output, output)


def get_video_path(device, l, s):
    base_path = "Hybrid Dataset/"
    paths = [base_path + "D{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*.mp4",
             base_path + "D{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*.MOV",
             base_path + "D{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*.mov",
             base_path + "D{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*.3gp",
             base_path + "D{:02d}".format(device) + "videos/outdoor/"
             ]
    result = glob.glob("Dataset/D" + "{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*.mp4")
    if not result:
        result = glob.glob(
            "Dataset/D" + "{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*.MOV")
    # assert len(result) == 1
    return result


def generate_video_sequence(sequence, l, s, output: str):
    print("Generating", output + ":")
    assert l != 0 and s != 0

    file_paths = [get_video_path(e, l, s)[0] if get_video_path(e, l, s) != [] else None for e in sequence]
    if None in file_paths:
        print("Skipping. Missing file")
        return
    if os.path.isfile(output):
        print("Skipping. Valid", output, "already exists")
        return
    concatenate_videos(file_paths, output)
    return


def main():
    for d in range(brand_devices_cnt):
        for l in range(1, locations_cnt + 1):
            for s in range(1, samples_cnt + 1):
                output = "output/Seq{:d}_Clip_L{:02d}S{:02d}.mp4".format(d + 1, l, s)
                generate_video_sequence(sequences[d], l, s, output)
    return


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    if len(sys.argv) == 4:
        d = int(sys.argv[1])
        l = int(sys.argv[2])
        s = int(sys.argv[3])
        output = "output/Seq{:d}_Clip_L{:02d}S{:02d}.mp4".format(d + 1, l, s)
        generate_video_sequence(sequences[d], l, s, output)
    else:
        main()
