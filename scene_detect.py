from scenedetect import detect, ContentDetector


def sequence_from_scenedetect(video_path):
    scene_list = detect(video_path, ContentDetector())
    assert len(scene_list) == 10
    return [e[1].frame_num for e in scene_list]
