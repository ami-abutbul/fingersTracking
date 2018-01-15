import platform
import os

#########################################################
# constants
#########################################################
image_height = 480
image_width = 640
warm_frames_num = 4

#########################################################
# parameters
#########################################################
rnn_state_size = 1024
batch_size = 1
learning_rate = 1e-5
epoch_num = 20000

restore_model = False
mode = 'train'  # 'train', 'test'

checkpoint_dir = 'models/model1/'
checkpoint_file = checkpoint_dir + 'model.ckpt'
log_dir = 'out.log'

cuda_visible_devices = '1'
if platform.system() == 'Linux':
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
