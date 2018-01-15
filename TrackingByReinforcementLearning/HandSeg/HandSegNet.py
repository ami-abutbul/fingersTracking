import sys
import threading

import numpy as np
import tensorflow as tf
from HandSeg.seg_colors import seg_colors, full_to_partial_seg
from PIL import Image
from Utilities.file_utils import dir_to_file_list, is_empty_dir, create_dir
from Utilities.cnn_utils import conv, deconv, batchnorm, conv_2_half_size
from HandSeg.configuration import *
from Utilities.data_set_handler import DataSetHandler, Dataset
from Utilities.image_utils import image_to_mat, mat_to_image, channel_to_image, concatenate_images_horizontally, \
    crop_random_tile, mask_to_rgb, partial_seg
from Utilities.producer_consumer_queue import ProducerConsumerQueue, QueueObj
from Utilities.stdout_log import StdoutLog

INPUTS = tf.placeholder(tf.float32, shape=[mb_size, image_size, image_size, 3], name='input')
MASKS = tf.placeholder(tf.int32, shape=[mb_size, image_size, image_size, 1], name='mask')


###########################################################################
#   Utilities functions for data handling.                                #
###########################################################################
lock = threading.Lock()
batch_index = 0


def next_batch_index(dataset_len):
    global lock, batch_index
    with lock:
        current_index = batch_index
        batch_index += (batch_index + mb_size) % dataset_len
        return current_index


def create_produce_obj_function():
    dataset = DataSetHandler(data_path, open_dataset, create_labels, pre_processing)

    def produce_obj():
        batch_idx = next_batch_index(dataset.len())
        return QueueObj([dataset.next_batch(mb_size, start_index=batch_idx)])
    return produce_obj


def open_dataset(dir_path):
    data_images = dir_to_file_list(os.path.join(dir_path, 'color'))
    data_masks = dir_to_file_list(os.path.join(dir_path, 'mask'))
    data_images.sort()
    data_masks.sort()
    return Dataset([data_images, data_masks])


def create_labels(input_path):
    image_path = input_path[0]
    mask_path = input_path[1]
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    return image, mask, None


def pre_processing(image, mask):
    cropped_images = crop_random_tile([image, mask], size=(image_size, image_size))
    img = image_to_mat(cropped_images[0])
    msk = image_to_mat(cropped_images[1], (image_size, image_size, 1))
    if add_noise:
        img = img + 0.5*img.std() * np.random.random(img.shape)
    if partial_seg:
        vec_partial_seg = np.vectorize(partial_seg)
        msk = vec_partial_seg(msk, full_to_partial_seg)
    return img / 255., msk


###########################################################################
#   Graph creation functions.                                             #
###########################################################################

def encoder(batch_input, reuse_variables):
    learned_variables = []
    stack_layers = []
    with tf.variable_scope("Encoder", reuse=reuse_variables) as encoder_scope:
        scope_name = encoder_scope.name

        conv_layer, conv_vars = conv(batch_input, 64, filter_size=3, scope_name=scope_name + '_conv_layer1')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm1')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer1') #image_size, image_size, 64
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 64, filter_size=3, scope_name=scope_name + '_conv_layer2')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm2')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer2') #image_size, image_size, 64
        learned_variables += conv_vars + batch_vars
        stack_layers.append(out)

        conv_layer, conv_vars = conv_2_half_size(out, 64, scope_name=scope_name + 'down_sampling1')
        out = tf.nn.relu(conv_layer, name=scope_name + 'down_sampling_relu1')#image_size/2, image_size/2, 64
        learned_variables += conv_vars

        conv_layer, conv_vars = conv(out, 128, filter_size=3, scope_name=scope_name + '_conv_layer3')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm3')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer3') #image_size/2, image_size/2, 128
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 128, filter_size=3, scope_name=scope_name + '_conv_layer4')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm4')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer4') #image_size/2, image_size/2, 128
        learned_variables += conv_vars + batch_vars
        stack_layers.append(out)

        conv_layer, conv_vars = conv_2_half_size(out, 128, scope_name=scope_name + 'down_sampling2')
        out = tf.nn.relu(conv_layer, name=scope_name + 'down_sampling_relu2') #image_size/4, image_size/4, 128
        learned_variables += conv_vars

        conv_layer, conv_vars = conv(out, 256, filter_size=3, scope_name=scope_name + '_conv_layer5')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm5')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer5') #image_size/4, image_size/4, 256
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 256, filter_size=3, scope_name=scope_name + '_conv_layer6')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm6')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer6') #image_size/4, image_size/4, 256
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 256, filter_size=3, scope_name=scope_name + '_conv_layer7')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm7')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer7') #image_size/4, image_size/4, 256
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 256, filter_size=3, scope_name=scope_name + '_conv_layer8')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm8')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer8') #image_size/4, image_size/4, 256
        learned_variables += conv_vars + batch_vars
        stack_layers.append(out)

        conv_layer, conv_vars = conv_2_half_size(out, 256, scope_name=scope_name + 'down_sampling3')
        out = tf.nn.relu(conv_layer, name=scope_name + 'down_sampling_relu3')#image_size/8, image_size/8, 256
        learned_variables += conv_vars

        conv_layer, conv_vars = conv(out, 512, filter_size=3, scope_name=scope_name + '_conv_layer9')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm9')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer9') #image_size/8, image_size/8, 512
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 512, filter_size=3, scope_name=scope_name + '_conv_layer10')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm10')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer10') #image_size/8, image_size/8, 512
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 512, filter_size=3, scope_name=scope_name + '_conv_layer11')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm11')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer11') #image_size/8, image_size/8, 512
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 512, filter_size=3, scope_name=scope_name + '_conv_layer12')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm12')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer12') #image_size/8, image_size/8, 512
        learned_variables += conv_vars + batch_vars
        stack_layers.append(out)

        conv_layer, conv_vars = conv_2_half_size(out, 512, scope_name=scope_name + 'down_sampling4')
        out = tf.nn.relu(conv_layer, name=scope_name + 'down_sampling_relu4')#image_size/16, image_size/16, 512
        learned_variables += conv_vars

        conv_layer, conv_vars = conv(out, 1024, filter_size=3, scope_name=scope_name + '_conv_layer13')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm13')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer13')  #image_size/16, image_size/16, 1024
        learned_variables += conv_vars + batch_vars

        return out, stack_layers, learned_variables


def decoder(batch_input, stack_layers):
    learned_variables = []
    with tf.variable_scope("Decoder") as decoder_scope:
        scope_name = decoder_scope.name

        conv_layer, conv_vars = deconv(batch_input, 512, filter_size=4, scope_name=scope_name + '_conv_layer1')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm1')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer1') #32, 32, 512
        learned_variables += conv_vars + batch_vars

        concat = tf.concat([out, stack_layers.pop()], axis=3) #32, 32, 1024
        conv_layer, conv_vars = conv(concat, 512, filter_size=3, scope_name=scope_name + '_conv_layer2')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm2')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer2') #32, 32, 512
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 512, filter_size=3, scope_name=scope_name + '_conv_layer3')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm3')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer3') #32, 32, 512
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 512, filter_size=3, scope_name=scope_name + '_conv_layer4')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm4')

        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer4') #32, 32, 512
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 512, filter_size=3, scope_name=scope_name + '_conv_layer5')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm5')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer5') #32, 32, 512
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = deconv(out, 256, filter_size=4, scope_name=scope_name + '_conv_layer6')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm6')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer6') #64, 64, 256
        learned_variables += conv_vars + batch_vars

        concat = tf.concat([out, stack_layers.pop()], axis=3)  # 64, 64, 512
        conv_layer, conv_vars = conv(concat, 256, filter_size=3, scope_name=scope_name + '_conv_layer7')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm7')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer7') #64, 64, 256
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 256, filter_size=3, scope_name=scope_name + '_conv_layer8')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm8')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer8') #64, 64, 256
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 256, filter_size=3, scope_name=scope_name + '_conv_layer9')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm9')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer9') #64, 64, 256
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 256, filter_size=3, scope_name=scope_name + '_conv_layer10')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm10')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer10') #64, 64, 256
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = deconv(out, 128, filter_size=4, scope_name=scope_name + '_conv_layer11')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm11')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer11') #128, 128, 128
        learned_variables += conv_vars + batch_vars

        concat = tf.concat([out, stack_layers.pop()], axis=3)  # 128, 128, 256
        conv_layer, conv_vars = conv(concat, 128, filter_size=3, scope_name=scope_name + '_conv_layer12')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm12')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer12')  # 128, 128, 128
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 128, filter_size=3, scope_name=scope_name + '_conv_layer13')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm13')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer13')  # 128, 128, 128
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = deconv(out, 64, filter_size=4, scope_name=scope_name + '_conv_layer14')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm14')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer14') #256, 256, 64
        learned_variables += conv_vars + batch_vars

        concat = tf.concat([out, stack_layers.pop()], axis=3)  # 256, 256, 128
        conv_layer, conv_vars = conv(concat, 64, filter_size=3, scope_name=scope_name + '_conv_layer15')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm15')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer15')  # 256, 256, 64
        learned_variables += conv_vars + batch_vars

        conv_layer, conv_vars = conv(out, 64, filter_size=3, scope_name=scope_name + '_conv_layer16')
        batch_norm, batch_vars = batchnorm(conv_layer, scope_name=scope_name + '_batch_norm16')
        out = tf.nn.relu(batch_norm, name=scope_name + '_relu_layer16')  # 256, 256, 64
        learned_variables += conv_vars + batch_vars

        out, conv_vars = conv(out, mask_depth, filter_size=1, scope_name=scope_name + '_conv_layer17')
        learned_variables += conv_vars + batch_vars

        return out, learned_variables


def hand_seg_net(input_batch):
    out_encoder, stack_layers, encoder_learned_variables = encoder(input_batch, False)
    out_masks, decoder_learned_variables = decoder(out_encoder, stack_layers)
    out_predicted_mask = tf.argmax(out_masks, axis=3)
    return out_predicted_mask, out_masks, out_encoder, encoder_learned_variables, decoder_learned_variables


###########################################################################
#   Execution functions.                                                  #
###########################################################################
def train():
    out_predicted_mask, out_masks, _, encoder_learned_vars, decoder_learned_vars = hand_seg_net(INPUTS)

    # reshape  input so that it becomes suitable for tf.softmax_cross_entropy_with_logits with [batch_size, num_classes]
    logit_masks = tf.reshape(out_masks, [-1, mask_depth])

    # create one hot tensor from labeled masks
    one_hot_labels = tf.reshape(tf.one_hot(MASKS, mask_depth, on_value=1.0, off_value=0.0, axis=-1), [-1, mask_depth])

    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_masks, labels=one_hot_labels)
    loss = tf.reduce_mean(entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,
                                           var_list=encoder_learned_vars + decoder_learned_vars)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # dataset = DataSetHandler(data_path, open_dataset, create_labels, pre_processing)
    batch_queue = ProducerConsumerQueue(create_produce_obj_function)

    with tf.Session() as sess:
        sess.run(init)

        if print_graph:
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            writer.close()

        if restore_model and not is_empty_dir(checkpoint_dir):
            saver.restore(sess, checkpoint_file)

        for i in range(1, epoch_num):
            # X_mb, Y_mb, _ = dataset.next_batch(mb_size)
            X_mb, Y_mb, _ = batch_queue.consume()
            _, curr_loss = sess.run([optimizer, loss], feed_dict={INPUTS: X_mb, MASKS: Y_mb})
            print("{}) loss: {}".format(i, curr_loss))

            if i % 500 == 0 and i != 0:
                saver.save(sess, checkpoint_file)

        saver.save(sess, checkpoint_dir)


def test():
    out_predicted_mask, _, _, _, _ = hand_seg_net(INPUTS)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    dataset = DataSetHandler(data_path, open_dataset, create_labels, pre_processing)

    with tf.Session() as sess:
        sess.run(init)

        if not is_empty_dir(checkpoint_dir):
            saver.restore(sess, checkpoint_file)

        for i in range(5):
            X_mb, Y_mb, _ = dataset.next_batch(mb_size)
            # mat_to_image(X_mb*255.).show()
            pred_mask = sess.run([out_predicted_mask], feed_dict={INPUTS: X_mb})
            pred_mask = mat_to_image(mask_to_rgb(np.array(pred_mask).reshape([-1]), seg_colors))

            mask = channel_to_image(Y_mb)
            concat = concatenate_images_horizontally([mask, pred_mask])
            concat.save('results/test_{}.png'.format(i))


def inference():
    out_predicted_mask, _, _, _, _ = hand_seg_net(INPUTS)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        if not is_empty_dir(checkpoint_dir):
            saver.restore(sess, checkpoint_file)

        import cv2
        cap = cv2.VideoCapture(0)
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frameP = np.reshape(cv2.resize(frame, (image_size, image_size)), [1, image_size, image_size, 3])/255.

            pred_mask = sess.run([out_predicted_mask], feed_dict={INPUTS: frameP})
            mat_to_image(mask_to_rgb(np.array(pred_mask).reshape([-1]), seg_colors)).show()
            mat_to_image(frame).show()

            # Display the resulting frame
            # cv2.imshow('frame', mask_to_rgb(np.array(pred_mask).reshape([-1]), seg_colors).reshape(256, 256, 3))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.stdout = StdoutLog(log_dir, sys.stdout, print_to_log, print_to_stdout)
    create_dir(checkpoint_dir)

    if mode == 'train':
        train()
    elif mode == 'test':
        test()
    else:
        inference()

