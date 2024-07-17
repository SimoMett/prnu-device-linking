import os

from scenedetect import detect, ContentDetector, AdaptiveDetector


def sequence_from_scenedetect(video_path):
    scene_list = detect(video_path, AdaptiveDetector(adaptive_threshold=2.73, weights=ContentDetector.Components(1.1, 1, 1, 1.3), min_scene_len=230))
    if not is_valid_seq(scene_list):
        raise RuntimeError("Invalid scenedetect sequence")
    return [0] + [e[1].frame_num for e in scene_list]


def is_valid_seq(scene_list):
    if len(scene_list) != 10:
        print("Length mismatch! Actual length:", len(scene_list))
        return False
    for s in scene_list:
        if not int(s[0].get_seconds()) % 10 == 0 and int(s[1].get_seconds()) % 10 == 0:
            print("Timeframes mismatch!")
            return False
    return True


if __name__ == "__main__":
    videos = ["output/" + str(v) for v in os.listdir("output")]
    for v in videos:
        print(v)
        if ".mp4" not in v:
            print(v, "is not a video")
            continue
        scene_list = detect(v, AdaptiveDetector(adaptive_threshold=2.73, weights=ContentDetector.Components(1.1, 1, 0, 1.3), min_scene_len=230, ))
        if len(scene_list) != 10:
            print("Length mismatch! Actual length:", len(scene_list))
            continue
        for s in scene_list:
            if not int(s[0].get_seconds()) % 10 == 0 and int(s[1].get_seconds()) % 10 == 0:
                print("Timeframes mismatch!")
                continue
