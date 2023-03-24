from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader


def setup_blip_model(model_path):
    
    blip_processor = AutoProcessor.from_pretrained(model_path)
    blip_model = BlipForConditionalGeneration.from_pretrained(model_path)
    
    return blip_processor, blip_model


def get_images_captions_from_blip(model, processor, image_paths, device, batch_size, gen_kwargs):
    
    model.eval()
    model.to(device)

    blip_data_loader = DataLoader(image_paths, batch_size=batch_size, shuffle=False)
    blip_captions = []

    for batch in tqdm(blip_data_loader):

        images = []
        for image_path in batch:
            i_image = Image.open(image_path).convert("RGB")
            images.append(i_image)
        
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        out = model.generate(pixel_values=pixel_values, **gen_kwargs)
        captions = processor.batch_decode(out, skip_special_tokens=True)

        blip_captions.extend(captions)
        
    # model.to("cpu")
    
    return blip_captions 