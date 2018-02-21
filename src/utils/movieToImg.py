import imageio
from PIL import Image

sourcePath = "../../VIDEO/MARATON/MP4/maraton.mp4"
destinationPath = "../../VIDEO/MARATON/IMG/"

if __name__ == "__main__":
    vid = imageio.get_reader(sourcePath, 'ffmpeg')

    for i, im in enumerate(vid):
        image = Image.fromarray(im, 'RGB')
        dest = destinationPath + str(i) + ".jpg"
        image.save(dest, 'JPEG')

