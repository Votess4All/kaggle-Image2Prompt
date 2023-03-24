import pandas as pd
import numpy as np
import sys 
sys.path.append('/kaggle/input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models


def set_up_stmodel(model_path="/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2"):
    st_model = SentenceTransformer(model_path)
    return st_model


def get_prompt_embeddings(comp_path, model_path="/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2"):
    prompts = pd.read_csv(comp_path / 'prompts.csv', index_col='imgId')
    prompts.head(7)

    # 7 * 384(feature_dim)这么多行
    sample_submission = pd.read_csv(comp_path / 'sample_submission.csv', index_col='imgId_eId')
    sample_submission.head()
    
    st_model = SentenceTransformer(model_path)
    prompt_embeddings = st_model.encode(prompts['prompt']).flatten()
    
    assert np.all(np.isclose(sample_submission['val'].values, prompt_embeddings, atol=1e-07))