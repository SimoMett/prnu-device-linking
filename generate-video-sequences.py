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


def get_clips_paths(device, base_path):
    paths = [base_path + "D{:02d}".format(device) + "_*/Nat/jpeg-h264/L*/S*/*",
             base_path + "D{:02d}".format(device) + "_*/videos/outdoor/*"
             ]
    file_formats = [".mp4", ".MOV", ".mov", ".3gp"]
    result = []
    for p in paths:
        for ff in file_formats:
            result += glob.glob(p+ff)
    return result


def generate_video_sequences(seq, max_index, base_path):
    for i in range(max_index):
        output_name = base_path + "output/Video_Seq" + str(sequences.index(seq)+1) + "_" + str(i)+".mp4"
        video_clips = []
        for dev in seq:
            paths = sorted(get_clips_paths(dev, base_path))
            video_clips.append(paths[i % len(paths)])
        concatenate_videos(video_clips, output_name)
        save_dataset_info(base_path, output_name, video_clips)
    return


def save_dataset_info(base_path, output_name, video_clips):
    with open("dataset_info.csv", "a") as f:
        f.write(",".join([output_name] + [v.removeprefix(base_path) for v in video_clips]) + "\n")


if __name__ == "__main__":
    os.makedirs("Hybrid Dataset/output", exist_ok=True)
    max_index = 34
    base_path = "Hybrid Dataset/"
    for seq in sequences:
        generate_video_sequences(seq, max_index, base_path)
