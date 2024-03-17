import json
import os

# container_rectangular_small
dim1 = (5.5, 6.3, 3.5)
bc1 = [
    (0, 0, 0),
    (0, 5.5, 0),
    (6.3, 5.5, 0),
    (6.3, 0, 0),
    (0, 0, 3.5),
    (0, 5.5, 3.5),
    (6.3, 5.5, 3.5),
    (6.3, 0, 3.5),
]

# container_rectangular_medium
dim2 = (12.8, 8.9, 12.8)
bc2 = [
    (0, 0, 0),
    (0, 8.9, 0),
    (12.8, 8.9, 0),
    (12.8, 0, 0),
    (0, 0, 12.8),
    (0, 8.9, 12.8),
    (12.8, 8.9, 12.8),
    (12.8, 0, 12.8),
]

# container_rectangular_large
dim3 = (19.0, 9.0, 27.9)
bc3 = [
    (0, 0, 0),
    (0, 9.0, 0),
    (19.0, 9.0, 0),
    (19.0, 0, 0),
    (0, 0, 27.9),
    (0, 9.0, 27.9),
    (19.0, 9.0, 27.9),
    (19.0, 0, 27.9),
]

# container_circular_small
dim4 = (11.0, 11.0, 6.6)
bc4 = [
    (0, 0, 0),
    (0, 11.5, 0),
    (11.5, 11.5, 0),
    (11.5, 0, 0),
    (0, 0, 6.6),
    (0, 11.5, 6.6),
    (11.5, 11.5, 6.6),
    (11.5, 0, 6.6),
]

# container_circular_medium
dim5 = (11.0, 11.0, 8.8)
bc5 = [
    (0, 0, 0),
    (0, 11.5, 0),
    (11.5, 11.5, 0),
    (11.5, 0, 0),
    (0, 0, 8.8),
    (0, 11.5, 8.8),
    (11.5, 11.5, 8.8),
    (11.5, 0, 8.8),
]

# container_circular_large
dim6 = (11.0, 11.0, 14.5)
bc6 = [
    (0, 0, 0),
    (0, 11.5, 0),
    (11.5, 11.5, 0),
    (11.5, 0, 0),
    (0, 0, 14.5),
    (0, 11.5, 14.5),
    (11.5, 11.5, 14.5),
    (11.5, 0, 14.5),
]

scene_dict = {
    1: {"class_id": 4, "world_offset": (13.2, 23.5, -6.6), "box_corners": bc4},
    2: {"class_id": 5, "world_offset": (13.2, 23.5, -8.8), "box_corners": bc5},
    3: {"class_id": 6, "world_offset": (13.2, 23.5, -14.5), "box_corners": bc6},
    4: {"class_id": 1, "world_offset": (18.2, 18.2, -3.5), "box_corners": bc1},
    5: {"class_id": 2, "world_offset": (20.0, 22.4, -12.8 ), "box_corners": bc2},
    6: {"class_id": 3, "world_offset": (24.6, 31.2,- 27.9), "box_corners": bc3},
    7: {"class_id": 4, "world_offset": (13.2, 23.5, -6.6), "box_corners": bc4},
    8: {"class_id": 5, "world_offset": (13.2, 23.5, -8.8), "box_corners": bc5},
    9: {"class_id": 6, "world_offset": (13.2, 23.5, -14.5), "box_corners": bc6},
    10: {"class_id": 1, "world_offset": (18.2, 18.2, -3.5), "box_corners": bc1},
    11: {"class_id": 2, "world_offset": (20.0, 22.4, -12.8 ), "box_corners": bc2},
    12: {"class_id": 3, "world_offset": (24.6, 31.2, -27.9), "box_corners": bc3},
    13: {"class_id": 4, "world_offset": (13.2, 23.5, -6.6), "box_corners": bc4},
    14: {"class_id": 5, "world_offset": (13.2, 23.5, -8.8), "box_corners": bc5},
    15: {"class_id": 6, "world_offset": (13.2, 23.5, -14.5), "box_corners": bc6},
    16: {"class_id": 1, "world_offset": (18.2, 18.2, -3.5), "box_corners": bc1},
    17: {"class_id": 2, "world_offset": (20.0, 22.4, -12.8 ), "box_corners": bc2},
    18: {"class_id": 3, "world_offset": (24.6, 31.2, -27.9), "box_corners": bc3},
    19: {"class_id": 4, "world_offset": (13.2, 23.5, -6.6), "box_corners": bc4},
    20: {"class_id": 5, "world_offset": (13.2, 23.5, -8.8), "box_corners": bc5},
    21: {"class_id": 6, "world_offset": (13.2, 23.5, -14.5), "box_corners": bc6},
    22: {"class_id": 1, "world_offset": (18.2, 18.2, -3.5), "box_corners": bc1},
    23: {"class_id": 2, "world_offset": (20.0, 22.4, -12.8 ), "box_corners": bc2},
    24: {"class_id": 3, "world_offset": (24.6, 31.2, -27.9), "box_corners": bc3},
    25: {"class_id": 4, "world_offset": (13.2, 23.5, -6.6), "box_corners": bc4},
    26: {"class_id": 5, "world_offset": (13.2, 23.5, -8.8), "box_corners": bc5},
    27: {"class_id": 6, "world_offset": (13.2, 23.5, -14.5), "box_corners": bc6},
    28: {"class_id": 1, "world_offset": (18.2, 18.2, -3.5), "box_corners": bc1},
    29: {"class_id": 2, "world_offset": (20.0, 22.4, -12.8 ), "box_corners": bc2},
    30: {"class_id": 3, "world_offset": (24.6, 31.2, -27.9), "box_corners": bc3},
    31: {"class_id": 4, "world_offset": (13.2, 23.5, -6.6), "box_corners": bc4},
    32: {"class_id": 5, "world_offset": (13.2, 23.5, -8.8), "box_corners": bc5},
    33: {"class_id": 6, "world_offset": (13.2, 23.5, -14.5), "box_corners": bc6},
    34: {"class_id": 1, "world_offset": (18.2, 18.2, -3.5), "box_corners": bc1},
    35: {"class_id": 2, "world_offset": (20.0, 22.4, -12.8 ), "box_corners": bc2},
    36: {"class_id": 3, "world_offset": (24.6, 31.2, -27.9), "box_corners": bc3},
    37: {"class_id": 4, "world_offset": (13.2, 23.5, -6.6), "box_corners": bc4},
    38: {"class_id": 5, "world_offset": (13.2, 23.5, -8.8), "box_corners": bc5},
    39: {"class_id": 6, "world_offset": (13.2, 23.5, -14.5), "box_corners": bc6},
    40: {"class_id": 1, "world_offset": (18.2, 18.2, -3.5), "box_corners": bc1},
    41: {"class_id": 2, "world_offset": (20.0, 22.4, -12.8 ), "box_corners": bc2},
    42: {"class_id": 3, "world_offset": (24.6, 31.2, -27.9), "box_corners": bc3},
    43: {"class_id": 4, "world_offset": (13.2, 23.5, -6.6), "box_corners": bc4},
    44: {"class_id": 5, "world_offset": (13.2, 23.5, -8.8), "box_corners": bc5},
    45: {"class_id": 6, "world_offset": (13.2, 23.5, -14.5), "box_corners": bc6},
    46: {"class_id": 1, "world_offset": (18.2, 18.2, -3.5), "box_corners": bc1},
    47: {"class_id": 2, "world_offset": (20.0, 22.4, -12.8 ), "box_corners": bc2},
    48: {"class_id": 3, "world_offset": (24.6, 31.2, -27.9), "box_corners": bc3},
    49: {"class_id": 4, "world_offset": (13.2, 23.5, -6.6), "box_corners": bc4},
    50: {"class_id": 5, "world_offset": (13.2, 23.5, -8.8), "box_corners": bc5},
    51: {"class_id": 6, "world_offset": (13.2, 23.5, -14.5), "box_corners": bc6},
    52: {"class_id": 1, "world_offset": (18.2, 18.2, -3.5), "box_corners": bc1},
    53: {"class_id": 2, "world_offset": (20.0, 22.4, -12.8 ), "box_corners": bc2},
    54: {"class_id": 3, "world_offset": (24.6, 31.2, -27.9), "box_corners": bc3},
}


if __name__ == "__main__":
    with open(os.path.join('./data', 'scene_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(scene_dict, f)

    