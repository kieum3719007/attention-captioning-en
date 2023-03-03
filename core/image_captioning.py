import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel, RobertaTokenizer
from transformers import ViTFeatureExtractor, TFViTModel
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
import os.path as osp
from core.train import train

def GetRobertaDecoder(pretrained): 
    configuration = RobertaConfig(is_decoder = True,
                                  add_cross_attention = True)
    model = TFRobertaModel(configuration)
    model.from_pretrained(pretrained)
    model.layers[0].submodules[1].trainable = False
    return model
    
def GetVitEncoder(pretrained_model):
    model = TFViTModel.from_pretrained(pretrained_model)
    model.layers[0].submodules[3].trainable = False
    return model

def GetViTPreprocess(pretrained_model):
    model = ViTFeatureExtractor.from_pretrained(pretrained_model)
    return model

def load_weight(model):
    cp = tf.train.Checkpoint(model=model,
                             optimizer=OPTIMIZER)
    cp_manager = tf.train.CheckpointManager(
        cp, CHECKPOINT_PATH, max_to_keep=5)
    cp.restore(cp_manager.latest_checkpoint)
    print(f'Loaded checkpoint from {CHECKPOINT_PATH}')

class TransformerCaptioner(Model):
    
    def __init__(self, config):
        super().__init__()

        self.image_preprocessor = GetViTPreprocess(config["pretrained_model"]["vit"])
        self.image_encoder = GetVitEncoder(config["pretrained_model"]["vit"])

        self.tokenizer = config["tokenizer"]
        self.decoder = GetRobertaDecoder(config["pretrained_model"]["roberta"])

        self.token_classifier = Dense(units=self.tokenizer.vocab_size)
    
    def call(self, image, text, training=False):        
        encoder_hidden_states = self.image_encoder(**image, training=training).last_hidden_state
        decoder_output = self.decoder(encoder_hidden_states=encoder_hidden_states, **text, training=training)
        output = self.token_classifier(decoder_output.last_hidden_state)
        return output    
    
CHECKPOINT_PATH =osp.join("model", "base-224")
TOKERNIZER = RobertaTokenizer.from_pretrained("roberta-base")
CONFIG = {
    "pretrained_model": {
        "vit": "google/vit-base-patch32-224-in21k",
        "roberta": "roberta-base"
    },
    "tokenizer": TOKERNIZER
}

def scce_with_ls(y, y_hat):
    y = tf.one_hot(tf.cast(y, tf.int32), TOKERNIZER.vocab_size)
    return categorical_crossentropy(y, y_hat, from_logits=True)

loss_object = scce_with_ls

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 1))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

LEARNING_RATE = 2e-5
LOSS = loss_function
OPTIMIZER = tf.keras.optimizers.legacy.Adam(LEARNING_RATE)

def create_model():
    model = TransformerCaptioner(CONFIG)
    model.compile(optimizer='adam', loss=LOSS)

    try:    
        load_weight(model)
        train(model, LOSS, OPTIMIZER)
    except:
        print("Load weight after train")
        load_weight(model)

    return model



    
