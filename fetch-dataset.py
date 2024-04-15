import urllib.request
import re
import requests
import os

general_regex = "_.*/Nat/jpeg-h264/L./S./.*(.MOV|.mp4)"
floreview_url = "https://lesc.dinfo.unifi.it/FloreView/"


def fetch_dataset(devices: list):
    for did in devices:
        assert type(did) is int

    # read all the urls
    content = urllib.request.urlopen(floreview_url + "FloreView_Dataset.txt").read().decode()

    # generate accepting regex and filter the correct files to download
    devices_formula = "D(" + "|".join(["{:02d}".format(d) for d in devices]) + ")"
    accepting_regex = devices_formula + general_regex
    url_list = list(filter(lambda s: re.search(accepting_regex, s) is not None, content.split("\n")))

    # make dirs and download the files
    for url in url_list:
        print("Downloading", url)
        file_dest = url.replace(floreview_url, "")
        if os.path.exists(file_dest):
            print("Skipping. Already exists")
            continue
        resp = requests.get(url)
        os.makedirs(os.path.dirname(file_dest), exist_ok=True)
        with open(file_dest, "wb") as f:
            f.write(resp.content)

    return


if __name__ == "__main__":
    fetch_dataset([2, 24, 19, 34, 5, 34, 40, 39, 32, 17, 38, 4])
