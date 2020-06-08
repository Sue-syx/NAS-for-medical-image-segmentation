config = {}
config['CUDA_VISIBLE_DEVICES'] = "0"
config['train_root'] = './dataset/train/'
config['augmentation_prob'] = 0.4
config['batch_size'] = 16
config['test_batch_size'] = 16
config['cuda'] = 'True'
config['loss_type'] = 'ce'
config['nclass'] = 1
config['filter_multiplier'] = 8
config['block_multiplier'] = 5
config['num_layers'] = 9
config['step'] = 5
config['lr'] = 0.025
config['momentum'] = 0.9
config['weight_decay'] = 1e-4
config['lr_scheduler'] = 'cos'
config['epochs'] = 200
config['min_lr'] = 0.0005
config['num_class'] = 2
config['steps'] = 4
config['up_cell_arch'] = [[0, 1], [1, 6], [1, 2], [1, 4]]
config['same_cell_arch'] = [[0, 4], [1, 2], [2, 5], [1, 1]]
config['down_cell_arch'] = [[0, 4], [1, 2], [2, 5], [1, 5]]
config['route_atch'] = [[0, 1, 0], [1, 2, 0], [2, 2, 2], [3, 2, 1], [4, 2, 2], [5, 2, 2], [6, 3, 0], [7, 3, 2], [8, 4, 0]]



