import urllib.request
import re
import requests
import os
import prnu_extract_fingerprints

floreview_regex = "_.*/Nat/jpeg-h264/L./S./.*(.MOV|.mp4|.3gp)"
floreview_url = "https://lesc.dinfo.unifi.it/FloreView/"
vision_url = "https://lesc.dinfo.unifi.it/VISION/"
vision_index = "VISION_files.txt"
vision_regex = "_.*/videos/(outdoor|indoor)/.*(.mov|.mp4|.3gp)"


def fetch_dataset(base_url: str, index_file: str, general_regex: str, devices: list, dest_folder: str):
    for did in devices:
        assert type(did) is int

    # read all the urls
    content = urllib.request.urlopen(base_url + index_file).read().decode()

    # generate accepting regex and filter the correct files to download
    devices_formula = "D(" + "|".join(["{:02d}".format(d) for d in devices]) + ")"
    accepting_regex = devices_formula + general_regex
    url_list = list(filter(lambda s: re.search(accepting_regex, s) is not None, content.split("\n")))

    # make dirs and download the files
    for url in url_list:
        print("Downloading", url)
        file_dest = url.replace(base_url, dest_folder)
        if os.path.exists(file_dest):
            print("Skipping. Already exists")
            continue
        resp = requests.get(url)
        os.makedirs(os.path.dirname(file_dest), exist_ok=True)
        with open(file_dest, "wb") as f:
            f.write(resp.content)

    return


if __name__ == "__main__":
    for s in prnu_extract_fingerprints.devs_sequences:
        fetch_dataset(floreview_url, "FloreView_Dataset.txt", floreview_regex, list(set(s)), "")
