import numpy as np
import tensorflow as tf
from core.image_captioning import TOKERNIZER


START = TOKERNIZER.cls_token_id
END = TOKERNIZER.eos_token_id





class Captioner(tf.Module):
  def __init__(self, transformer):
    self.transformer = transformer

  def __call__(self, image, max_length=80):
    # input sentence is portuguese, hence adding the start and end token
    image_input = self.transformer.image_preprocessor(image, return_tensors="tf")
    input_ids = tf.convert_to_tensor([[self.transformer.tokenizer.cls_token_id]], dtype=tf.int64)
    single_mask = tf.convert_to_tensor([[1]], dtype=tf.int64)
    attention_mask = tf.convert_to_tensor([[1]], dtype=tf.int64)
    
    for i in tf.range(max_length):
      text_input = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
        }
      predictions = self.transformer(image_input, text_input)
      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
      predicted_id = tf.argmax(predictions, axis=-1)

      if(predicted_id.numpy()[0][0]==self.transformer.tokenizer.eos_token_id):
        break
      
      attention_mask = tf.concat([attention_mask, single_mask], axis=-1)

      input_ids = tf.concat([input_ids, predicted_id], axis=-1)
      
        
    output = input_ids[0][1:]
    tokens = self.transformer.tokenizer.convert_ids_to_tokens(output)
    text = self.transformer.tokenizer.convert_tokens_to_string(tokens)
    return text