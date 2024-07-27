import os

from scenedetect import detect, ContentDetector, AdaptiveDetector


def sequence_from_scenedetect(video_path):
    detectors = [AdaptiveDetector(adaptive_threshold=2.73, min_scene_len=240),
                 AdaptiveDetector(adaptive_threshold=2.73, weights=ContentDetector.Components(1.1, 1, 0, 1.3),
                                  min_scene_len=240),
                 AdaptiveDetector(adaptive_threshold=2.6, min_scene_len=240),
                 ContentDetector(min_scene_len=240)]
    for detector in detectors:
        scene_list = detect(video_path, detector)
        if is_valid_seq(scene_list):
            return [0] + [e[1].frame_num for e in scene_list]
    raise RuntimeError("Invalid scenedetect sequence")


def is_valid_seq(scene_list):
    if len(scene_list) != 10:
        print("Length mismatch! Actual length:", len(scene_list))
        return False
    for s in scene_list:
        if int(s[0].get_seconds()) % 10 != 0 and int(s[1].get_seconds()) % 10 != 0:
            print("Timeframes mismatch!")
            return False
    return True


def verify_on_dataset():
    videos = ["Hybrid Dataset/output/" + str(v) for v in os.listdir("Hybrid Dataset/output")]
    for v in sorted(videos):
        if ".mp4" not in v:
            continue
        print(v)
        detectors = [AdaptiveDetector(adaptive_threshold=2.73, min_scene_len=240),
                     AdaptiveDetector(adaptive_threshold=2.73, weights=ContentDetector.Components(1.1, 1, 0, 1.3),
                                      min_scene_len=240),
                     AdaptiveDetector(adaptive_threshold=2.6, min_scene_len=240),
                     ContentDetector(min_scene_len=240)]
        for i, detector in enumerate(detectors):
            scene_list = detect(v, detector)
            if not is_valid_seq(scene_list):
                continue
            print("Best detector:", i)
            break


if __name__ == "__main__":
    verify_on_dataset()
