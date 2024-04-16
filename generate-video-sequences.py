import ffmpeg
import glob

locations_cnt = 7
scenes_cnt = 5  # used to be 6, but dataset is lacking

#             A   G  A  H  A   M  A   S  A   X
sequence_a = (2, 19, 2, 5, 2, 40, 2, 32, 2, 38)


# extract clip WITHOUT audio, but for the purpose of this project it's fine
def extract_clip(input_file: str, output: str, start=0, duration=10):
    (
        ffmpeg
        .input(input_file)
        .trim(start=start, duration=duration)
        .output(output)
        .run()
    )


def concatenate_videos(videos: list, output: str):
    clips = []
    for v in videos:
        if v is not None:
            clips.append(ffmpeg.input(v).trim(start=0, end=4))
    (
        ffmpeg
        .concat(*clips)
        .output(output)
        .run()
    )


def generate_videos(sequence, i, j):
    assert i != 0 and j != 0

    print("Generating sequence:")
    file_paths = [get_video_path(e, i, j)[0] if get_video_path(e, i, j) != [] else None for e in sequence]
    concatenate_videos(file_paths, "test.mp4")
    return


def get_video_path(device, l, s):
    result = glob.glob("Dataset/D" + "{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*")
    assert len(result) == 1
    return result


def main():
    generate_videos(sequence_a, 1, 2)

    return


if __name__ == "__main__":
    main()
