import os
import re
import binascii
import PIL.Image as Image
import io
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup


def download_images(src="https://www.gettyimages.com/photos/office-workers-wearing-face-masks?mediatype=photography&phrase=office%20workers%20wearing%20face%20masks&sort=mostpopular"):
    req = Request(src, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(req).read()
    bs_obj = BeautifulSoup(html, "html.parser")

    imgs = bs_obj.findAll("img", {"src": re.compile(r"http*")})
    imgs_urls = []

    for img in imgs:
        try:
            # print(img["src"])
            imgs_urls.append(img["src"]) # extracting the urls, avoiding encountering images without src's
        except Exception as e:
            pass

    return imgs_urls


def read_and_download(url, img_num, filename="./mask_images/"):
    """
    read one url and download the image, saving as a .png file in ./mask_images directory
    """
    try:
        img_data = urlopen(url).read()
        # print(img_data)
        # b_data = binascii.unhexlify(img_data)
        # img = Image.open(io.BytesIO(b_data))
        img = Image.open(io.BytesIO(img_data))
        path = os.path.join(filename, f"img_{img_num}.png")
        img.save(path)
    except Exception as e:
        print(e)
        
            
def run_image_downloader():
    # # store the urls into a .txt file
    # # once the file is written, please comment this part and uncomment "read the urls..." part and run it separately
    # max_num_imgs = 450
    # imgs_out = download_images() # the first page
    #
    # for i in range(2, 10):
    #     if len(imgs_out) > max_num_imgs:
    #         break
    #
    #     print(f"current: page {i}")
    #     src = f"https://www.gettyimages.com/photos/office-workers-wearing-face-masks?mediatype=photography&page={i}" + \
    #           f"&phrase=office%20workers%20wearing%20face%20masks&sort=mostpopular" # substitute "&page=... part"
    #
    #     imgs_out += download_images(src=src)
    #     print(f"current number of images: {len(imgs_out)}")
    #
    # print(f"final number of images: {len(imgs_out)}")
    #
    # with open("./masked_imgs_urls.txt", "w") as wf:
    #     for url in imgs_out:
    #         wf.write(f"{url}\n")

    # read the urls and download the images
    with open("./masked_imgs_urls.txt", "r") as rf:
        ind = 1
        url = rf.readline().strip()
        while len(url) > 0:
            # print(f"current url: {url}")
            if ind % 10 == 0:
                print(f"current ind: {ind}")

            read_and_download(url, ind)
            ind += 1

            url = rf.readline().strip()


def read_and_ravel_imgs(max_num_imgs=480, src="./lfw", tgt="./face_images"):
    pattern = r".*\.jpg"
    ind = 1
    break_flag = False
    for root, dirs, files in os.walk(src):
        if break_flag:
            break

        for filename in files:
            matched = re.match(pattern, filename)
            if bool(matched):
                image = Image.open(os.path.join(root, filename))
                image.save(os.path.join(tgt, f"img_{ind}.png"))
                ind += 1
                if ind % 10 == 0:
                    print(f"current ind: {ind}")
                if ind > max_num_imgs:
                    break_flag = True
                    break
            break # only stores one image for one person
            
