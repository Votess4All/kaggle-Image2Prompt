from transformers import AutoProcessor, BlipForConditionalGeneration


def setup_blip_model(model_path="/kaggle/input/blip-pretrained-model/blip-image-captioning-large"):
    
    blip_processor = AutoProcessor.from_pretrained(model_path)
    blip_model = BlipForConditionalGeneration.from_pretrained(model_path)
    
    return blip_processor, blip_model