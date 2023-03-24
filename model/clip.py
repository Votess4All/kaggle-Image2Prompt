import open_clip


def set_up_clip_model(model_name, model_path, device):
    
    clip_model = open_clip.create_model(model_name, precision="fp16" if device == "cuda" else "fp32")
    open_clip.load_checkpoint(clip_model, model_path)
    clip_tokenizer = open_clip.get_tokenizer(model_name)
    clip_preprocess = open_clip.image_transform(
        clip_model.visual.image_size,
        is_train = False,
        mean = getattr(clip_model.visual, 'image_mean', None),
        std = getattr(clip_model.visual, 'image_std', None),
    )
    
    return clip_model, clip_tokenizer, clip_preprocess