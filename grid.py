from PIL import Image

def make_grid(img_folder:str, frames: int, n: int):
    skip = (n+1)*4
    video_d = open('data.txt', "r")
    boxes = list(map(lambda x: list(map(lambda x: int(x), x.split())), video_d.readlines()))
    width = max(map(lambda x: x[2], boxes))
    height = max(map(lambda x: x[3], boxes))
    while (width * 2) % 8 != 0:
        width += 1
    while (height * 2) % 8 != 0:
        height += 1
    for i in range(frames // skip):
        image1 = Image.open(f"{img_folder}/face_box/0_frame{i * skip}.jpg")
        image2 = Image.open(f"{img_folder}/face_box/0_frame{i * skip + (n+1)}.jpg")
        image3 = Image.open(f"{img_folder}/face_box/0_frame{i * skip + (n+1)*2}.jpg")
        image4 = Image.open(f"{img_folder}/face_box/0_frame{i * skip + (n+1)*3}.jpg")
        mask1 = Image.open(f"{img_folder}/mask/0_frame{i * skip}.jpg")
        mask2 = Image.open(f"{img_folder}/mask/0_frame{i * skip + (n+1)}.jpg")
        mask3 = Image.open(f"{img_folder}/mask/0_frame{i * skip + (n+1)*2}.jpg")
        mask4 = Image.open(f"{img_folder}/mask/0_frame{i * skip + (n+1)*3}.jpg")

        comb = Image.new("RGB", (width*2, height*2))
        comb.paste(image1, (0,0))
        comb.paste(image2, (width,0))
        comb.paste(image3, (0,height))
        comb.paste(image4, (width,height))
        comb.save(f"{img_folder}/grid/frames{i}.jpg")

        comb_m = Image.new("RGB", (width*2, height*2))
        comb_m.paste(mask1, (0,0))
        comb_m.paste(mask2, (width,0))
        comb_m.paste(mask3, (0,height))
        comb_m.paste(mask4, (width,height))
        comb_m.save(f"{img_folder}/grid_mask/frames{i}.jpg")


def de_grid(img_folder:str, frames: int, n: int):
    skip = (n+1)*4
    video_d = open('data.txt', "r")
    boxes = list(map(lambda x: list(map(lambda x: int(x), x.split())), video_d.readlines()))
    width = max(map(lambda x: x[2], boxes))
    height = max(map(lambda x: x[3], boxes))
    while (width * 2) % 8 != 0:
        width += 1
    while (height * 2) % 8 != 0:
        height += 1
    for i in range(frames // skip):
        image1 = Image.open(f"{img_folder}/face_box/0_frame{i * skip}.jpg")
        image2 = Image.open(f"{img_folder}/face_box/0_frame{i * skip + (n+1)}.jpg")
        image3 = Image.open(f"{img_folder}/face_box/0_frame{i * skip + (n+1)*2}.jpg")
        image4 = Image.open(f"{img_folder}/face_box/0_frame{i * skip + (n+1)*3}.jpg")
        if i >= 10 and i < 100:
            ai = f"000{i}-frames{i}"
        elif i >= 100:
            ai = f"00{i}-frames{i}"
        elif i < 10:
            ai = f"0000{i}-frames{i}" 
        grid = Image.open(f"{img_folder}/grid_ai/{ai}.png")
        ai1 = grid.crop((0, 0, image1.width, image1.height))
        ai2 = grid.crop((width, 0, width+image2.width, image2.height))
        ai3 = grid.crop((0, height, image3.width, height+image3.height))
        ai4 = grid.crop((width, height, width+image4.width, height+image4.height))
        ai1.save(f"{img_folder}/ai/frame{i * skip}.png")
        ai2.save(f"{img_folder}/ai/frame{i * skip + (n+1)*1}.png")
        ai3.save(f"{img_folder}/ai/frame{i * skip + (n+1)*2}.png")
        ai4.save(f"{img_folder}/ai/frame{i * skip + (n+1)*3}.png")