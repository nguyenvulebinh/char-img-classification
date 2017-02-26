import os

import sys
from PIL import Image, ImageDraw, ImageFont
from os import listdir
from os.path import isfile, join

image_default_size = 28
delta = 2
font_folder = '/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/font-brift-custom/font-brift'
out_folder = '/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/data_1'


def get_index(character):
    if character.isdigit():
        return character
    elif character.lower() == character:
        return str(ord(character) - ord('a') + 36)
    else:
        return str(ord(character) - ord('A') + 10)


def generate_image(font_type, character):
    base = Image.open('bg28.png').convert('RGBA')
    txt = Image.new('RGBA', base.size, (255, 255, 255, 0))
    font_size = 10
    try:
        fnt = ImageFont.truetype(font_type, font_size)
    except Exception:
        return
    d = ImageDraw.Draw(txt)

    while fnt.getsize(character)[0] < image_default_size and fnt.getsize(character)[1] < image_default_size:
        font_size += 1
        fnt = ImageFont.truetype(font_type, font_size)

    if 10 <= int(get_index(character)) <= 35:
        # draw text, full opacity
        if fnt.getsize(character)[0] < fnt.getsize(character)[1]:
            d.text(((image_default_size - fnt.getsize(character)[0]) / 2, -2), character, font=fnt,
                   fill=(255, 255, 255, 255))
        else:
            d.text((0, (image_default_size - fnt.getsize(character)[1]) / 2 - 2), character, font=fnt,
                   fill=(255, 255, 255, 255))

    else:
        # draw text, full opacity
        if fnt.getsize(character)[0] < fnt.getsize(character)[1]:
            d.text(((image_default_size - fnt.getsize(character)[0]) / 2, - delta), character, font=fnt,
                   fill=(255, 255, 255, 255))
        else:
            d.text((0, (image_default_size - fnt.getsize(character)[1]) / 2 - delta), character, font=fnt,
                   fill=(255, 255, 255, 255))

    out = Image.alpha_composite(base, txt).convert('L')

    try:
        out.save(out_folder + os.sep + "data" + os.sep + get_index(character) + os.sep + str(
            (font_type[font_type.rindex(os.sep) + 1:-4])) + ".png")
    except Exception:
        os.mkdir(out_folder + os.sep + "data" + os.sep + get_index(character))
        out.save(out_folder + os.sep + "data" + os.sep + get_index(character) + os.sep + str(
            (font_type[font_type.rindex(os.sep) + 1:-4])) + ".png")

    print(font_type + ": " + character)


# try:
#     font_folder = str(sys.argv[1])
#     out_folder = str(sys.argv[2])
# except Exception:
#     print("command <folder font> <folder output>")
#     exit()
try:
    os.mkdir(out_folder + "data")
    print("Already to generate")
except Exception:
    print("Already to generate")

onlyfiles = [join(font_folder, f) for f in listdir(font_folder) if isfile(join(font_folder, f))]
for str_font_path in onlyfiles:
    if str_font_path.endswith(".TTF") or str_font_path.endswith(".ttf"):
        for i in range(ord('A'), ord('Z') + 1):
            generate_image(str_font_path, chr(i))
        for i in range(ord('a'), ord('z') + 1):
            generate_image(str_font_path, chr(i))
        for i in range(0, 10):
            generate_image(str_font_path, str(i))
