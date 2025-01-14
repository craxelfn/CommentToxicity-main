import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
import gradio as gr

# Load the data
df = pd.read_csv(os.path.join('jigsaw-toxic-comment-classification-challenge','train.csv', 'train.csv'))

# Setup vectorizer
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                             output_sequence_length=1800,
                             output_mode='int')

# Adapt vectorizer
X = df['comment_text']
vectorizer.adapt(X.values)

# Load model
model = tf.keras.models.load_model('toxicity.h5')

# Scoring function
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text

# Create and launch interface
interface = gr.Interface(
    fn=score_comment, 
    inputs=gr.Textbox(lines=2, placeholder='Comment to score'),  # Updated syntax
    outputs='text'
)

if __name__ == "__main__":
    interface.launch(share=True)