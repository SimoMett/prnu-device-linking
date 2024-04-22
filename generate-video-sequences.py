import ffmpeg
import glob
import os


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
    default_duration = 1

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
    ffmpeg.concat(*[c.filter("scale", "1920-1080").filter("setsar", "1-1") for c in clips]).output(output).overwrite_output().run()


def generate_video_sequence(sequence, l, s, output: str):
    print("Generating", output+":")
    assert l != 0 and s != 0

    file_paths = [get_video_path(e, l, s)[0] if get_video_path(e, l, s) != [] else None for e in sequence]
    concatenate_videos(file_paths, output)
    return


def get_video_path(device, l, s):
    result = glob.glob("Dataset/D" + "{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*.mp4")
    if not result:
        result = glob.glob("Dataset/D" + "{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*.MOV")
    # assert len(result) == 1
    return result


locations_cnt = 7
scenes_cnt = 4  # used to be 6, but dataset is lacking
brand_devices_cnt = 2
devices_pairs = [
    (2, 24),
    (19, 34),
    (5, 33),
    (40, 39),
    (32, 17),
    (38, 4)
]
sequences = [
    #(2, 19, 2, 5, 2, 40, 2, 32, 2, 38),
    (13, 19, 13, 5, 13, 40, 13, 32, 13, 38),
    #(24, 34, 24, 33, 24, 39, 24, 17, 24, 4)
    (35, 34, 35, 33, 35, 39, 35, 17, 35, 4)
    # A   G   A   H   A   M  A   S   A   X
]


def main():
    os.makedirs("output", exist_ok=True)
    for d in range(brand_devices_cnt):
        for l in range(1, locations_cnt + 1):
            for s in range(1, scenes_cnt + 1):
                output = "output/Seq{:d}_Clip_L{:02d}S{:02d}.mp4".format(d + 1, l, s)
                generate_video_sequence(sequences[d], l, s, output)
    return


def debug_main():
    l = 1
    s = 1
    d = 1
    output = "output/Seq{:d}_Clip_L{:02d}S{:02d}.mp4".format(d + 1, l, s)
    generate_video_sequence(sequences[d], l, s, output)


if __name__ == "__main__":
    debug_main()
