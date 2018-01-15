import platform
import os

#########################################################
# constants
#########################################################
EPS = 1e-12
output_channels = 3
mask_depth = 8
image_size = 256

#########################################################
# DRQN parameters
#########################################################
actions_num = 38
units_size = 1024       # number of lstm units, also the size of the last conv layer in thw DRQN
update_freq = 5         # How often to perform a training step.
gamma = .99             # Discount factor on the target Q-values
startE = 1              # Starting chance of random action
endE = 0.1              # Final chance of random action
max_trace_len = 50      # The max allowed length of trace
anneling_steps = 10000  # How many steps of training to reduce startE to endE.
pre_train_steps = 10000 # How many steps of random actions before training begins.
tau = 0.001


#########################################################
# parameters
#########################################################
mb_size = 1
epoch_num = 20000
learning_rate = 1e-5

mode = 'infer'
train_path = 'segHandData/train' if platform.system() == 'Linux' else 'D:/private/datasets/RHD_v1-1/RHD_published_v2/training'
test_path = 'segHandData/test' if platform.system() == 'Linux' else 'D:/private/datasets/RHD_v1-1/RHD_published_v2/evaluation'
data_path = train_path if mode == 'train' else test_path

restore_model = True
add_noise = True
partial_seg = True
print_to_log = True
print_to_stdout = True
print_graph = False

checkpoint_dir = 'models/model4/'
checkpoint_file = checkpoint_dir + 'model.ckpt'
log_dir = 'out.log'

cuda_visible_devices = '1'
if platform.system() == 'Linux':
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
