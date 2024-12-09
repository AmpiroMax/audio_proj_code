from PIL import Image


def get_custom_metadata(info, audio):
    path = "/home/mpatratskiy/work/audio_proj/audio_proj_code/data/images/vivaldi_autumn_chunk_1.jpeg"
    image = Image.open(path)
    return {"prompt": [image]}
