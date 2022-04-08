#!/usr/bin/env python3

import requests
import os
import cfbm


here = cfbm.__file__.split("__init__.py")[0]

# Need to replace with download from service we go with (e.g. CANFAR)
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in list(response.cookies.items()):
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def main():
    print("Downloading CHIME/FRB Beam Model")
    file_ids = {
        "beam_XX_v1.h5": "",
        "beam_YY_v1.h5": "",
    }
    for filename in file_ids.keys():
        print("Fetching: {}".format(filename))
        directory = here + "/bm_data/"
        if not os.path.exists(directory):
            print("Making beam model data directory at {}...".format(directory))
        destination = directory + "{}".format(filename)
        if not os.path.isfile(destination):
            download_file_from_google_drive(file_ids[filename], destination)
    print("Download Complete")


if __name__ == "__main__":
    main()
