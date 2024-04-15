devices_list = [
    ("D22_Apple_iPadAir", "D24_Apple"),
    ("D19_Google_Pixel3a", "D34_Google_Pixel5"),
    ("D05_Huawei_P9Lite", "D33_Huawei_Mate10Pro"),
    ("D40_Motorola_MotoG9Plus", "D39_Motorola_MotoG5"),
    ("D32_Samsung_GalaxyA52s", "D17_Samsung_GalaxyS21+"),
    ("D38_Xiaomi_Redmi5Plus", "D04_Xiaomi_RedmiNote8T")
]
devices_list_plain = [
    "D22_Apple_iPadAir", "D24_Apple",
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
scenes_cnt = 6


def generate_videos(device, i, j):
    #for dev in devices_list_plain:
        #print(dev + "/L" + str(i + 1) + "/S" + str(j + 1) + "/")
    print(device + "/L" + str(i + 1) + "/S" + str(j + 1) + "/")
    return


def main():
    for i in range(locations_cnt):
        for j in range(scenes_cnt):
            for dev in devices_list_plain:
                generate_videos(dev, i, j)
    return


if __name__ == "__main__":
    main()
