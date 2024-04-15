import ffmpeg
import os
import glob

devices_list = [
    ("D02_Apple_iPhoneX", "D24_Apple_iPadAir"),
    ("D19_Google_Pixel3a", "D34_Google_Pixel5"),
    ("D05_Huawei_P9Lite", "D33_Huawei_Mate10Pro"),
    ("D40_Motorola_MotoG9Plus", "D39_Motorola_MotoG5"),
    ("D32_Samsung_GalaxyA52s", "D17_Samsung_GalaxyS21+"),
    ("D38_Xiaomi_Redmi5Plus", "D04_Xiaomi_RedmiNote8T")
]
devices_list_plain = [
    "D02_Apple_iPhoneX", "D24_Apple_iPadAir",
    "D19_Google_Pixel3a", "D34_Google_Pixel5",
    "D05_Huawei_P9Lite", "D33_Huawei_Mate10Pro",
    "D40_Motorola_MotoG9Plus", "D39_Motorola_MotoG5",
    "D32_Samsung_GalaxyA52s", "D17_Samsung_GalaxyS21+",
    "D38_Xiaomi_Redmi5Plus", "D04_Xiaomi_RedmiNote8T"
]
devices_list2 = [
    (22, 24),
    (19, 34),
    (5, 33),
    (40, 39),
    (32, 17),
    (38, 4)
]
locations_cnt = 7
scenes_cnt = 5  # used to be 6, but dataset is lacking

#              A   G   A  H   A   M   A   S   A   X
sequence_a = (2, 19, 2, 5, 2, 40, 2, 32, 2, 38)


def generate_videos(sequence, i, j):
    assert i != 0 and j != 0

    print("Concatenating the following videos:")
    [print(get_video_path(e, i, j)) for e in sequence]
    return


def get_video_path(device, l, s):
    result = glob.glob("Dataset/D" + "{:02d}".format(device) + "_*/Nat/jpeg-h264/L" + str(l) + "/S" + str(s) + "/*")
    # assert result != []
    return result


def main():
    generate_videos(sequence_a, 1, 2)

    return


if __name__ == "__main__":
    main()
