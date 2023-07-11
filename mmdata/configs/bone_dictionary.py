

bone_jp_eng_dictionary = {
    "上半身": "Torso",
    "左手首": "Left-Hand",
    "右手首": "Right-Hand",
    "左足首": "Left-Foot",
    "右足首": "Right-Foot",
    "左ひざ": "Left-Knee",
    "右ひざ": "Right-Knee",
    "左ひじ": "Left-Elbow",
    "右ひじ": "Right-Elbow",
    "左肩": "Left-Shoulder",
    "右肩": "Right-Shoulder",
    "左足": "Left-Hip",
    "右足": "Right-Hip",
    "頭": "Head",
    "首": "Neck",
    "左目": "Left-Eye",
    "右目": "Right-Eye",
}
bone_jp_eng_dictionary.update({v: k for (k, v) in bone_jp_eng_dictionary.items()})

open_pose_format = {
    "Head": 0,
    "Neck": 1,
    "Right-Shoulder": 2,
    "Right-Elbow": 3,
    "Right-Hand": 4,
    "Left-Shoulder": 5,
    "Left-Elbow": 6,
    "Left-Hand": 7,
    "Right-Hip": 9,
    "Right-Knee": 10,
    "Right-Foot": 11,
    "Left-Hip": 12,
    "Left-Knee": 13,
    "Left-Foot": 14,
    "Right-Eye": 15,
    "Left-Eye": 16,
}
