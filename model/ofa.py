from tqdm import tqdm
from PIL import Image
from transformers import OFATokenizer, OFAModel
from torch.utils.data import DataLoader
# from transformers.models.ofa.generate import sequence_generator

def setup_ofa_model(model_path, device):
    tokenizer = OFATokenizer.from_pretrained(model_path)
    model = OFAModel.from_pretrained(model_path, use_cache=False).to(device)
    
    return tokenizer, model


def get_images_captions_from_ofa(model, processor, image_paths, device, batch_size, gen_kwargs):
    
    model.eval()
    model.to(device)

    data_loader = DataLoader(image_paths, batch_size=batch_size, shuffle=False)
    captions = []

    for batch in tqdm(data_loader):

        images = []
        for image_path in batch:
            i_image = Image.open(image_path).convert("RGB")
            images.append(i_image)
        

        out = model.generate(inputs.repeat(len(img_batch), 1).cuda(), patch_images=img_batch, num_beams=5, no_repeat_ngram_size=2)
        out_captions = tokenizer.batch_decode(out, skip_special_tokens=True)
        out_captions = [cap + ", fine details, masterpiece" for cap in out_captions]

        
        captions.extend(out_captions)
        
    # model.to("cpu")
    
    return captions 