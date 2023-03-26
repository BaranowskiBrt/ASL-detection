# Keypoints taken from https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts

silhouette_keypoints = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
rightEyeUpper1 = [247, 30, 29, 27, 28, 56, 190]
rightEyeLower1 = [130, 25, 110, 24, 23, 22, 26, 112, 243]
rightEyeUpper2 = [113, 225, 224, 223, 222, 221, 189]
rightEyeLower2 = [226, 31, 228, 229, 230, 231, 232, 233, 244]
rightEyeLower3 = [143, 111, 117, 118, 119, 120, 121, 128, 245]

rightEyebrowUpper = [156, 70, 63, 105, 66, 107, 55, 193]
rightEyebrowLower = [35, 124, 46, 53, 52, 65]

# rightEyeIris = [473, 474, 475, 476, 477]

leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
leftEyeUpper1 = [467, 260, 259, 257, 258, 286, 414]
leftEyeLower1 = [359, 255, 339, 254, 253, 252, 256, 341, 463]
leftEyeUpper2 = [342, 445, 444, 443, 442, 441, 413]
leftEyeLower2 = [446, 261, 448, 449, 450, 451, 452, 453, 464]
leftEyeLower3 = [372, 340, 346, 347, 348, 349, 350, 357, 465]

leftEyebrowUpper = [383, 300, 293, 334, 296, 336, 285, 417]
leftEyebrowLower = [265, 353, 276, 283, 282, 295]

# leftEyeIris = [468, 469, 470, 471, 472]

midwayBetweenEyes = [168]

noseTip = [1]
noseBottom = [2]
noseRightCorner = [98]
noseLeftCorner = [327]

rightCheek = [205]
leftCheek = [425]

all_lips = lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner
all_eyes = (
    rightEyeUpper0
    + rightEyeLower0
    + rightEyeUpper1
    + rightEyeLower1
    + rightEyeUpper2
    + rightEyeLower2
    + rightEyeLower3
    # + rightEyeIris
    + leftEyeUpper0
    + leftEyeLower0
    + leftEyeUpper1
    + leftEyeLower1
    + leftEyeUpper2
    + leftEyeLower2
    + leftEyeLower3
    # + leftEyeIris
)
all_eyebrows = rightEyebrowUpper + rightEyebrowLower + leftEyebrowUpper + leftEyebrowLower

pose_keypoints = list(range(489, 521 + 1))
left_hand_keypoints = list(range(468, 488 + 1))
right_hand_keypoints = list(range(522, 542 + 1))


def extract_keypoints(
    silhouette: bool = False,
    lips: bool = False,
    eyes: bool = False,
    eyebrows: bool = False,
    rest_of_face: bool = False,
    pose: bool = False,
    left_hand: bool=False,
    right_hand: bool=False,
):
    keypoints = []
    if silhouette:
        keypoints += silhouette_keypoints
    if lips:
        keypoints += all_lips
    if eyes:
        keypoints += all_eyes
    if eyebrows:
        keypoints += all_eyebrows
    if rest_of_face:
        keypoints += list(
            set(range(1, 468)) - set(silhouette_keypoints + all_lips + all_eyes + all_eyebrows)
        )
    if pose:
        keypoints += pose_keypoints
    if left_hand:
        keypoints += left_hand_keypoints
    if right_hand:
        keypoints += right_hand_keypoints

    return list(set(keypoints))
