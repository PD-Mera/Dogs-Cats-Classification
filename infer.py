import os
import torch
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

from model import init_model
from config import *


def draw_to_image(image: Image, text: str):
    draw = ImageDraw.Draw(image)
    h, w = image.size
    size = min(h, w) // 5
    font = ImageFont.truetype('asset/arial.ttf', size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font = font)
    draw.rectangle((left-5, top-5, right+5, bottom+5), fill="black")
    draw.text((0, 0), text, (255,255,255), font = font)
    return image


def test(config):
    model = init_model(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(config['result_path']):
        os.mkdir(config['result_path'])

    if os.path.isdir(config['path']):  
        print(f"Starting infer {len(os.listdir(config['path']))} images")
        for imagelink in tqdm(os.listdir(config['path'])):
            if os.path.isdir(os.path.join(config['path'], imagelink)):
                print(f"{os.path.join(config['path'], imagelink)} is a folder. Exiting..")
                exit()
            image = Image.open(os.path.join(config['path'], imagelink))
            inputs = transform(image).unsqueeze(0).to(device)
            results = torch.argmax(model(inputs)).cpu()
            classname = config['class']['name'][results]
            image = draw_to_image(image, classname)
            image.save(os.path.join(config['result_path'], imagelink))
        print(f"Image saved to {config['result_path']}")
         
    elif os.path.isfile(config['path']):  
        print(f"Starting infer 1 images")
        image = Image.open(config['path'])
        inputs = transform(image).unsqueeze(0).to(device)
        results = torch.argmax(model(inputs)).cpu()
        classname = config['class']['name'][results]
        image = draw_to_image(image, classname)
        image.save(os.path.join(config['result_path'], config['path'].split('/')[-1]))
        print(f"Image saved to {os.path.join(config['result_path'], config['path'].split('/')[-1])}")
    else:  
        print("Save path invalid")
        exit()

if __name__ == '__main__':
    test(TESTING__CFG)