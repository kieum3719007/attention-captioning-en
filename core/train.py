import tensorflow as tf
import numpy as np
import os.path as osp

BATCH_SIZE = 4
IMAGE_INPUT_SHAPE = (3, 224, 224)
train_step_signature = [
    tf.TensorSpec(shape=(BATCH_SIZE, *IMAGE_INPUT_SHAPE), dtype=tf.float32),
    tf.TensorSpec(shape=(BATCH_SIZE, 40), dtype=tf.int32),
    tf.TensorSpec(shape=(BATCH_SIZE, 40), dtype=tf.int32),
]


def train(model, _loss, optimizer):

    CKPT = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)
    CHECKPOINT_PATH = osp.join("model", "base-224")
    CKPT_MANAGER = tf.train.CheckpointManager(
        CKPT, CHECKPOINT_PATH, max_to_keep=5)

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, seq, mask):

        tar_inp = {
            "input_ids": seq[:, :-1],
            "attention_mask": mask[:, :-1]
        }

        tar_real = {
            "input_ids": seq[:, 1:],
            "attention_mask": mask[:, 1:]
        }

        inp = {
            "pixel_values": inp
        }

        with tf.GradientTape() as tape:
            predictions = model(inp, tar_inp,
                                        training=True)
            loss = _loss(tar_real["input_ids"], predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss
    
    for epoch in range(1):
        total_loss = 0
        total_step = 0
        print("\nStart of epoch %d" % (epoch + 1,))
        for step in range(1):
            # print(inp/.shape)
            inp = np.random.uniform(0, 1, (BATCH_SIZE, *IMAGE_INPUT_SHAPE))
            seq =  np.random.randint(1, 100, size = (BATCH_SIZE, 40))
            mask =  np.random.randint(0, 1, size = (BATCH_SIZE, 40))
            loss_value = train_step(inp, seq, mask)
            if step % 50 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
            total_loss += loss_value
            total_step += 1

        print("Loss of epoch %d is %.4f" % (epoch + 1, total_loss/total_step))  

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = CKPT_MANAGER.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')