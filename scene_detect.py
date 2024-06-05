import os

from scenedetect import detect, ContentDetector, AdaptiveDetector


def sequence_from_scenedetect(video_path):
    scene_list = detect(video_path, AdaptiveDetector(adaptive_threshold=2.73, weights=ContentDetector.Components(1.1, 1, 0, 1.3)))
    return [0] + [e[1].frame_num for e in scene_list]


if __name__ == "__main__":
    videos = ["output/" + str(v) for v in os.listdir("output")]
    videos.remove("output/Seq2_Clip_L04S03.mp4")  # FIXME this video is not rendered correctly
    for v in videos:
        print(v)
        if ".mp4" not in v:
            print(v, "is not a video")
            continue
        scene_list = detect(v, AdaptiveDetector(adaptive_threshold=2.73, weights=ContentDetector.Components(1.1, 1, 0, 1.3)))
        if len(scene_list) != 10:
            print("Length mismatch! Actual length:", len(scene_list))
            continue
        for s in scene_list:
            if not int(s[0].get_seconds()) % 10 == 0 and int(s[1].get_seconds()) % 10 == 0:
                print("Timeframes mismatch!")
                continue
