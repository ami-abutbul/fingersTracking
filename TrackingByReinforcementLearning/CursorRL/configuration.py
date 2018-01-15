import platform
import os

#########################################################
# constants
#########################################################
EPS = 1e-12
output_channels = 3
mask_depth = 8
image_height = 480
image_width = 640

#########################################################
# DRQN parameters
#########################################################
actions_num = 74
units_size = 512        # number of lstm units, also the size of the last conv layer in thw DRQN
update_freq = 5         # How often to perform a training step.
gamma = 0.5             # Discount factor on the target Q-values
startE = 1              # Starting chance of random action
endE = 0.05             # Final chance of random action
max_trace_len = 8       # The max allowed length of trace
max_study_len = 150     # The max allowed length of trace
annealing_steps = 1000  # How many steps of training to reduce startE to endE.
pre_train_steps = 1000  # How many steps of random actions before training begins.
update_target_ratio = 0.001
warm_frames_num = 2

#########################################################
# cursor parameters
#########################################################
step_size = 20
good_radius = 8             # max distance to all points which considered as close to target.
ok_radius = 16              # max distance to all points which considered as close enough to target, but not good.
sec_waiting_for_click = 1   # how many sec (max) we should wait between getting to clickable point and clicking
rewards = [1.5,             # PLACE_GOOD
           0.5,             # PLACE_OK
           -3,              # PLACE_BAD
           1.5,             # CLICK_GOOD
           -3,              # CLICK_BAD
           -1,              # CLICK_MISSING
           1]               # WAITING


# For drqn_v2
# rewards = [1,             # PLACE_GOOD
#            0.33,          # PLACE_OK
#            0.01,          # PLACE_BAD
#            1,             # CLICK_GOOD
#            0.01,          # CLICK_BAD
#            0.01,          # CLICK_MISSING
#            1]             # WAITING

#########################################################
# parameters
#########################################################
mb_size = 1
epoch_num = 30000
learning_rate = 1e-5

mode_type = 'fingers'  # 'hand' , 'fingers'
mode = "test"  # 'test' , 'train'

restore_model = False
print_to_log = True
print_to_stdout = True
print_graph = False
encoder_checkpoint_file = "../HandSeg/models/model4/model.ckpt"


if mode_type == 'fingers':
    stats_dir = "stats/fingers" if mode == "train" else "stats/test/fingers"
    checkpoint_dir = 'models/fingers_models/model1_r8-16/'
    checkpoint_file = checkpoint_dir + 'model.ckpt'
    log_dir = 'finger_out.log'

    cuda_visible_devices = '1'
    if platform.system() == 'Linux':
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

else:
    stats_dir = "stats/hands" if mode == "train" else "stats/test/hands"
    checkpoint_dir = 'models/full_hands_models/model1_r8-16/'
    checkpoint_file = checkpoint_dir + 'model.ckpt'
    log_dir = 'hands_out.log'

    cuda_visible_devices = '1'
    if platform.system() == 'Linux':
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

# def set_run_config(run_mode_type):
#     global stats_dir, checkpoint_dir, checkpoint_file, log_dir, mode_type
#     mode_type = run_mode_type
#     if run_mode_type == 'fingers':
#         stats_dir = "stats/fingers"
#         checkpoint_dir = 'models/fingers_models/model1_r8-16/'
#         checkpoint_file = checkpoint_dir + 'model.ckpt'
#         log_dir = 'finger_out.log'
#
#         cuda_visible_devices = '0'
#         if platform.system() == 'Linux':
#             os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
#
#     else:
#         stats_dir = "stats/hands"
#         checkpoint_dir = 'models/full_hands_models/model1_r8-16/'
#         checkpoint_file = checkpoint_dir + 'model.ckpt'
#         log_dir = 'hands_out.log'
#
#         cuda_visible_devices = '1'
#         if platform.system() == 'Linux':
#             os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

# set_run_config(mode_type)
