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
    default_duration = 3

    # For each video sources I have to trim the video and create a split filter using 'filter_multi_output'
    # and then count every stream occurrence used during encoding process to avoid using the same stream multiple times
    # (otherwise I'll get the error "ValueError: encountered trim with multiple outgoing edges...")
    videos_set = set(videos)  # no duplicates
    splits_map = dict.fromkeys(videos_set)
    for v in splits_map.keys():
        if v is not None:
            splits_map[v] = (
                [ffmpeg.input(v).trim(start=default_start, end=default_duration).filter_multi_output("split"), 0])
        else:
            splits_map[v] = (
                [ffmpeg.input(videos[1]).trim(start=default_start, end=default_duration).filter_multi_output("split"),
                 0])  # FIXME

    # Now I build the clips sequence allocating the streams of each sources
    clips = []
    for v in videos:
        if v is not None:
            clips.append(splits_map[v][0].stream(splits_map[v][1]))
            splits_map[v][1] += 1
        else:
            clips.append(splits_map[videos[1]][0].stream(splits_map[videos[1]][1]))
            splits_map[videos[1]][1] += 1

    # Finally, generate output
    ffmpeg.concat(*clips).output(output).overwrite_output().run()


def generate_video_sequence(sequence, l, s, output):
    assert l != 0 and s != 0

    print("Generating sequence:")
    file_paths = [get_video_path(e, l, s)[0] if get_video_path(e, l, s) != [] else None for e in sequence]
    concatenate_videos(file_paths, output)
    return


def get_video_path(device, l, s):
    result = glob.glob("Dataset/D" + "{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*")
    # assert len(result) == 1
    return result


locations_cnt = 7
scenes_cnt = 5  # used to be 6, but dataset is lacking
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
    (2,  19,  2,  5,  2, 40,  2, 32,  2, 38),
    (24, 34, 24, 33, 24, 39, 24, 17, 24, 4)
#     A   G   A   H   A   M  A   S   A   X
]


def main():
    os.makedirs("output", exist_ok=True)
    for l in range(1, locations_cnt + 1):
        for s in range(1, scenes_cnt + 1):
            for d in range(brand_devices_cnt):
                output = "output/Seq{:d}_Clip_L{:02d}S{:02d}.mp4".format(d+1, l, s)
                print("Generating", output)
                generate_video_sequence(sequences[d], l, s, output)
    return


if __name__ == "__main__":
    main()
