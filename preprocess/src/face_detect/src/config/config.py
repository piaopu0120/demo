def init_config():
    cfg_mnet = {
        'name': 'mobilenet0.25',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'image_size': 640,
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 64,
        'weights_path':'../src/face_detect/src/weights/mobilenet0.25_Final.pth',
        'onet_weights_path':'../src/face_detect/src/weights/onet.npy', # for tracker
        'device': 'cuda',
        'target_size': 640,
        'keep_size': False,
        'onet_threshold': 0.6, # for tracker
        'target_size_tracker': 320, # for tracker
    }
    return cfg_mnet
