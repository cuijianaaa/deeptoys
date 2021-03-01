import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
from datetime import datetime
from easydict import EasyDict as edict
import torch

class Logger():
    def __init__(self,
                 tb_log_dir='./results/tflog',
                 tb_filter='',
                 tb_log_hist=False,
                 separator_len=100):
        self.tb_log_hist = tb_log_hist
        try:
            from tensorboardX import SummaryWriter
            print('Use tensorboardX as logger.')
            self.tb_logger = SummaryWriter(tb_log_dir)
        except:
            print('WARNING: Can\'t import tensorboardX, you can install by:\n'
                  'pip install tensorboardX && pip install tensorboard')
            self.tb_log_hist = False
            self.tb_logger = None

        self.tb_filter = tb_filter

        self.total_time = 0.0
        self.num_runs = 0

        self.losses = {}
        self.metrics = {}

        self.separator_len = separator_len

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        diff_time = time.time() - self.start_time
        self.total_time += diff_time
        self.num_runs += 1
        self.average_time = self.total_time / self.num_runs
        return diff_time 
    
    def add_loss(self, name, loss):
        self.losses[name] = loss.mean(dim=0, keepdim=True)

    def add_metric(self, name, metric):
        self.metrics[name] = metric
    
    def total_loss(self):
        total_loss = 0
        for k, loss in self.losses.items():
            total_loss += loss
        return total_loss

    def log_train(self, epoch, global_step, lr, num_steps, step, num_steps_per_epoch, params_dict):
        def _log_to_terminal():
            eta_seconds = self.average_time  * (num_steps - global_step)
            hours, remainder = divmod(eta_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            lines = \
                '[Epoch {:d} ({:d}/{:d})][Step {:d} ({:.2f}%)][Lr {}][Time {:.4f}s]'\
                '[ETA {:d}h{:d}m{:d}s][{}]\n'.format(
                    epoch + 1, step + 1, num_steps_per_epoch, global_step,
                    global_step * 100.0 / num_steps, lr, self.average_time, int(hours),
                    int(minutes), int(seconds), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            separator = '-' * self.separator_len + '\n'
            lines += separator
            lines += ('{:>' + str(self.separator_len // 2) + '} = {:<.6f}\n').format(
                'total_loss', self.total_loss())
            lines += separator
            for k, v in self.losses.items():
                lines += ('{:>' + str(self.separator_len // 2) + '} = {:<.6f}\n').format(k, v)
                lines += separator
            print(lines)

        def _log_to_tensorboard():
            for k, loss in self.losses.items():
                name = 'training_losses/{}'.format(k)
                self.tb_logger.add_scalar(name, loss.item(), global_step)
            if self.tb_log_hist:
                for k, v in parmas_dict:
                    if self.tb_filter in k:
                        self.tb_tblogger.add_histogram(k, v, global_step)

        _log_to_terminal()
        if self.tb_logger:
            _log_to_tensorboard()

    def log_test(self, global_step, log_to_tensorboard=True):
        def _log_to_terminal():
            title = 'Evaluation:'
            separator = '*' * self.separator_len + '\n'
            left_len = (self.separator_len - len(title)) // 2
            right_len = self.separator_len - left_len - len(title)
            lines = '*' * left_len + title + '*' * right_len + '\n'
            for k, v in self.metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                lines += ('{:>' + str(self.separator_len // 2) + '} = {:<.6f}\n').format(k, v)
                lines += separator
            print(lines)

        def _log_to_tensorboard(): 
            for k, metric in self.metrics.items():
                name = 'test_metrics/{}'.format(k)
                if isinstance(metric, torch.Tensor):
                    metric = metric.item()
                self.tb_logger.add_scalar(name, metric, global_step)
        
        _log_to_terminal()
        if self.tb_logger and log_to_tensorboard:
            _log_to_tensorboard()

class Table():
    '''
    Table with field_names, for example:
    
    ['input', 'output']
    ['rgb_image1', 'class_probs1']
    ['rgb_image2', 'class_probs2']

    
    # Create table:
    table = Table(['input', 'output'])

    # Clear table:
    table.clear()

    # Add row:
    table.add(['rgb_image1', 'class_probs1'])  # add by list
    table.add({'input': 'rgb_image2', 'output': 'class_prob2'})  # add by dict

    # Iterate over table:
    for line in table:
        print(type(line))
        print(line)
    # output:
    # <class 'dict'>
    # {'input': 'rgb_image2', 'output': 'class_prob2'}

    '''

    def __init__(self, field_names=None):
        self.field_names = field_names
        if field_names is not None:
            assert isinstance(field_names, list), '\'field_names\' must be list of string!'
        self.table = []

    def clear(self):
        self.table = []

    def add(self, input):
        if isinstance(input, list):
            assert self.field_names is not None, \
                '\'field_names\' must be set before add row by list!'
            assert len(self.field_names) == len(input), \
                'input length mismatch with field_name length {} vs {}'.format(
                    len(input), len(self.field_names))
        elif isinstance(input, dict):
            if self.field_names is None:
                assert len(self.table) == 0, 'table should not be set before field_names be set!'
                self.field_names = list(sorted(input.keys()))
            input = [input[field_name] for field_name in self.field_names]
        else:
            raise ValueError('Input type must be list or dict, but {}'.format(type(input)))

        assert len(self.field_names) == len(input), \
            'input length mismatch with field_name length {} vs {}'.format(
                len(input), len(self.field_names))
        self.table.append(input)

    def __len__(self):
        return len(self.table)

    def __iter__(self):
        self.iter = iter(self.table)
        return self

    def __next__(self):
        if self.field_names is None:
            raise StopIteration
        return dict(zip(self.field_names, next(self.iter)))

class Pipeline():
    '''
    class of train/test/deploy pipeline
    '''
    def __init__(self, args):
        self.cfg = self._load_cfg(args.cfg)
        self.pipe_cfg = self.cfg.PIPELINE
        assert args.mode in ['train', 'test', 'deploy'], \
            'Unknow mode {}, must be train, test or deploy.'.format(args.mode)
        self.mode = args.mode
        self.train_batch_size = self.pipe_cfg.TRAIN.BATCH_SIZE
        self.test_batch_size = self.pipe_cfg.TEST.BATCH_SIZE 
        self.logger = Logger(
            tb_log_dir=self.pipe_cfg.TRAIN.RESULTS_ROOT + '/tflog',
            tb_filter=self.pipe_cfg.TRAIN.TB_FLITER,
            tb_log_hist=self.pipe_cfg.TRAIN.TB_LOG_PARAM_HIST)
        self._eval_table = Table()
        self.device_settings = self._set_device(args.cuda)
        self.device = self.device_settings['device']
        print('Load dataloader...')
        self.train_data_loader, self.test_data_loader = self._load_data()
        print('Load model...')
        self.model_cfg, self.model = self._load_model()
        print('Load optimizer...')
        self.optimizer = self._load_optimizer()
        print('Load weight...')
        self._load_weight()
        print('Set parallel...')
        self._set_parallel()
        print('Start run...')
        self.global_step = 0
        if args.mode == 'train':
            self._run_train()
        elif args.mode == 'test':
            self._run_test()
        else:
            self._run_deploy()

    def load_default_model_cfg(self):
        '''
        Override this function if has model config, it will be called by Pipeline automatically.
        '''
        return edict()
    
    def loss(self, data, output):
        '''
        Override this to define a loss function, it will be called by Pipeline automatically.
        data: data directly from data loader
        output: model output
        '''
        NotImplementedError

    def add_loss(self, name, loss):
        '''
        Call this function in self.loss() to collect the losses to Pipeline.
        The added loss will be used to:
            1. optimize the network
            2. log to terminal
            3. log to tensorboard
        '''
        self.logger.add_loss(name, loss)

    def eval_step(self, data, output):
        '''
        Override this function to save per step test result to self._eval_table, it will be used in
        self.eval(), eval_step will run in every test step
        data: data directly from data loader
        output: model output
        '''
        NotImplementedError

    def add_eval(self, input):
        '''
        Call this function in eval_step to save per step test result.
        '''
        self._eval_table.add(input)

    def eval(self):
        '''
        Override this function to do evaluation after testing over all test data examples, you can
        get all test result by calling self.eval_table(), after evaluation, call self.add_metric()
        to collect evaluation metric to Pipeline.
        '''
        NotImplementedError

    def eval_table(self):
        '''
        Call this function to get all test result.
        '''
        return self._eval_table

    def add_metric(self, name, metric):
        '''
        Call this function in self.eval() to collect evaluation metric to Pipeline.
        The added metric will be used to:
            1. log to terminal
            2. log to tensorboard
        '''
        self.logger.add_metric(name, metric)

    def _load_default_pipe_cfg(self):
        # ============================Pipeline config============================
        PIPELINE = edict()

        # data config
        PIPELINE.DATA = edict()
        PIPELINE.DATA.ROOT = ''
        PIPELINE.DATA.SCRIPT = 'kitti_object'
        PIPELINE.DATA.LOADER = 'kittiObjectDataset'
        PIPELINE.DATA.NUM_WORKERS = 1

        # model config
        PIPELINE.MODEL = edict()
        PIPELINE.MODEL.TASK = 'detection'
        PIPELINE.MODEL.SCRIPT = ''
        PIPELINE.MODEL.NAME = ''

        # training config
        PIPELINE.TRAIN = edict()
        PIPELINE.TRAIN.BATCH_SIZE = 4
        PIPELINE.TRAIN.START_EPOCH = 0
        PIPELINE.TRAIN.END_EPOCH = 15
        
        PIPELINE.TRAIN.OPTIMIZER = 'adam'
        PIPELINE.TRAIN.LEARNING_RATE = 0.00005
        PIPELINE.TRAIN.WEIGHT_DECAY = 0.0005
        PIPELINE.TRAIN.DOUBLE_BIAS = False
        PIPELINE.TRAIN.BIAS_DECAY = False
        PIPELINE.TRAIN.MOMENTUM = 0.9
        PIPELINE.TRAIN.LR_DECAY_GAMMA = 0.1
        PIPELINE.TRAIN.LR_DECAY_EPOCH = 10

        PIPELINE.TRAIN.CLIP_GRAD = -1

        PIPELINE.TRAIN.PRETRAINED = False
        PIPELINE.TRAIN.PRETRAINED_WEIGHT = '' 
        PIPELINE.TRAIN.RESUME = False
        PIPELINE.TRAIN.RESUME_EPOCH = 0
        PIPELINE.TRAIN.FINE_TUNE = False
        PIPELINE.TRAIN.FINE_TUNE_STRICT = False
        PIPELINE.TRAIN.FINE_TUNE_MODEL = ''
        PIPELINE.TRAIN.FINE_TUNE_STATE_DICT = True
        PIPELINE.TRAIN.SNAPSHOT_INTERVAL = 10000
        PIPELINE.TRAIN.RESULTS_ROOT = 'results'
        PIPELINE.TRAIN.SNAPSHOT_PREFIX = 'tinyyolo'

        PIPELINE.TRAIN.EVAL_INTERVAL = -1
        PIPELINE.TRAIN.DISP_INTERVAL = 200

        PIPELINE.TRAIN.TB_FLITER = ''
        PIPELINE.TRAIN.TB_LOG_PARAM_HIST = False

        # test config
        PIPELINE.TEST = edict()
        PIPELINE.TEST.BATCH_SIZE = 1
        PIPELINE.TEST.EVAL_SCRIPT = 'eval.evaluate'
        PIPELINE.TEST.EVAL_FUNCTION = 'evaluate'
        PIPELINE.TEST.WRITE_CACHE = True
        PIPELINE.TEST.EVAL_USE_CACHE = False
        PIPELINE.TEST.TEST_EPOCH = 15
        PIPELINE.TEST.LOAD_STATE_DICT = False
        PIPELINE.TEST.VIS = False

        # deploy config
        PIPELINE.DEPLOY = edict()
        PIPELINE.DEPLOY.ONNX2NCNN = '/home/cj/repos/ncnn/build/tools/onnx/onnx2ncnn'

        return PIPELINE

    def _load_default_cfg(self):
        CONFIG = edict()
        CONFIG.PIPELINE = self._load_default_pipe_cfg()
        CONFIG.MODEL = self.load_default_model_cfg()
        return CONFIG
    
    def _load_user_cfg(self, default_cfg, cfg_file):
        def __override_default_cfg(user_cfg, default_cfg):
            """
            Override default config by user config if user config exists.
            """
            if type(user_cfg) is not edict:
                return
            for key, value in user_cfg.items():
                # item in user_cfg must in default_cfg
                if key not in default_cfg:
                    raise KeyError('{} is not user_cfg valid config key'.format(key))
                # the types must match, too
                old_type = type(default_cfg[key])
                if old_type is not type(value):
                    if isinstance(default_cfg[key], np.ndarray):
                        value = np.array(value, dtype=default_cfg[key].dtype)
                    else:
                        raise ValueError(('Type mismatch ({} vs. {}) '
                                          'for config key: {}').format(type(default_cfg[key]),
                                                                       type(value), key))
                # recursively override
                if type(value) is edict:
                    try:
                        __override_default_cfg(user_cfg[key], default_cfg[key])
                    except:
                        print(('Error under config key: {}'.format(k)))
                        raise
                else:
                    default_cfg[key] = value
        
        import yaml
        with open(cfg_file, 'r') as f:
            # Use safe_load instead of load to avoid unnecessary warning
            yaml_cfg = edict(yaml.safe_load(f))
        __override_default_cfg(yaml_cfg, default_cfg)
        return default_cfg # overrided

    def _load_cfg(self, cfg_file):
        default_cfg = self._load_default_cfg()
        final_cfg = self._load_user_cfg(default_cfg, cfg_file)
        return final_cfg

    def _set_device(self, use_cuda):
        device_dict = {}
        if use_cuda:
            cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
            gpu_list = cuda_visible_devices.split(',')
            visible_num_gpus = len(gpu_list)
            device_dict['visible_num_gpus'] = visible_num_gpus
            print('Use GPU devices')
            print('Available GPU stats:')
            print('number: {} ids: {}'.format(visible_num_gpus, cuda_visible_devices))
            assert torch.cuda.is_available(), 'There is no gpu device available, try cpu.'
            device = torch.device('cuda')
        else:
            print('Use CPU device')
            device = torch.device('cpu')
        device_dict['device'] = device
        return device_dict

    def _load_data(self):
        exec('from dataset.' + self.pipe_cfg.DATA.SCRIPT + ' import ' + self.pipe_cfg.DATA.LOADER)

        if self.mode == 'train':
            train_dataset = eval(self.pipe_cfg.DATA.LOADER)(
            os.path.join(os.path.dirname(__file__), 'data', self.pipe_cfg.DATA.ROOT), train=True)
            collate_fn = getattr(train_dataset, 'collate_fn', None)
            # todo delete this if after update pytorch? 
            if collate_fn is not None:
                train_data_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=self.pipe_cfg.DATA.NUM_WORKERS,
                    collate_fn=collate_fn)
            else:
                train_data_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=self.pipe_cfg.DATA.NUM_WORKERS)
        else:
            train_data_loader = None

        test_dataset = eval(self.pipe_cfg.DATA.LOADER)(
            os.path.join(os.path.dirname(__file__), 'data', self.pipe_cfg.DATA.ROOT), train=False)
        collate_fn = getattr(test_dataset, 'collate_fn', None)
        # todo delete this if after update pytorch? 
        if collate_fn is not None:
            test_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.pipe_cfg.DATA.NUM_WORKERS,
                collate_fn=collate_fn)
        else:
            test_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.pipe_cfg.DATA.NUM_WORKERS)

        return train_data_loader, test_data_loader

    def _load_model(self):
        exec('from model.' + self.pipe_cfg.MODEL.TASK + '.' + self.pipe_cfg.MODEL.SCRIPT \
            + ' import ' + self.pipe_cfg.MODEL.NAME)
        model_cfg = self.cfg.MODEL
        model = eval(self.pipe_cfg.MODEL.NAME)(model_cfg)
        model = model.to(self.device_settings['device'])
        return model_cfg, model

    def _load_optimizer(self):
        LR = self.pipe_cfg.TRAIN.LEARNING_RATE
        WEIGHT_DECAY = self.pipe_cfg.TRAIN.WEIGHT_DECAY
        DOUBLE_BIAS = self.pipe_cfg.TRAIN.DOUBLE_BIAS
        BIAS_DECAY = self.pipe_cfg.TRAIN.BIAS_DECAY

        params = []
        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    lr = LR * (2 if DOUBLE_BIAS else 1)
                    weight_decay = WEIGHT_DECAY if BIAS_DECAY else 0
                else:
                    lr = LR
                    weight_decay = WEIGHT_DECAY
                    
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

        if self.pipe_cfg.TRAIN.OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(params)
        elif self.pipe_cfg.TRAIN.OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(params, momentum=self.pipe_cfg.TRAIN.MOMENTUM)
        else:
            raise Exception('Not supported optimizer, should be sgd or adam')
        return optimizer

    def _load_weight(self):
        SNAPSHOT_DIR = os.path.join(self.pipe_cfg.TRAIN.RESULTS_ROOT, 'snapshots')
        if self.mode == 'train':
            if not os.path.isdir(SNAPSHOT_DIR):
                os.makedirs(SNAPSHOT_DIR)
            if self.pipe_cfg.TRAIN.RESUME:
                self.pipe_cfg.TRAIN.FINE_TUNE = False
                print('Resuming training stats, fine_tune will be disabled')
                load_name = os.path.join(SNAPSHOT_DIR, self.pipe_cfg.TRAIN.SNAPSHOT_PREFIX) + \
                            '_{}.pth'.format(self.pipe_cfg.TRAIN.RESUME_EPOCH)
                print("Resuming checkpoint {}".format(load_name))
                checkpoint = torch.load(load_name)
                self.pipe_cfg.TRAIN.START_EPOCH = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['model'], strict=True)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # update init lr if resume
                self.pipe_cfg.TRAIN.LEARNING_RATE = self.optimizer.param_groups[0]['lr']
                print("loaded checkpoint {}".format(load_name))
            else:
                assert self.pipe_cfg.TRAIN.START_EPOCH == 0, \
                    'epoch should start from 0 if not resume'

            if self.pipe_cfg.TRAIN.FINE_TUNE:
                print("Finetuning checkpoint from {}".format(self.pipe_cfg.TRAIN.FINE_TUNE_MODEL))
                if self.pipe_cfg.TRAIN.FINE_TUNE_STATE_DICT:
                    ft_checkpoint = torch.load(self.pipe_cfg.TRAIN.FINE_TUNE_MODEL)
                else:
                    ft_checkpoint = torch.load(self.pipe_cfg.TRAIN.FINE_TUNE_MODEL)['model']
                self.model.load_state_dict(ft_checkpoint, 
                                           strict=self.pipe_cfg.TRAIN.FINE_TUNE_STRICT)
        elif self.mode == 'test' or 'deploy':
            load_name = os.path.join(SNAPSHOT_DIR, self.pipe_cfg.TRAIN.SNAPSHOT_PREFIX) + '_{}.pth'\
                .format(self.pipe_cfg.TEST.TEST_EPOCH)
            print("load checkpoint {} for {}".format(load_name, self.mode))
            checkpoint = torch.load(load_name)
            if self.pipe_cfg.TEST.LOAD_STATE_DICT:
                self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint['model'])
            print('load model successfully!')
        else:
            print('WARNING: load weight in {} mode is not implemented!'.format(self.mode))

    def _set_parallel(self):
        if 'visible_num_gpus' in self.device_settings \
            and self.device_settings['visible_num_gpus'] > 1:
            self.model = torch.nn.DataParallel(self.model)

    def _save_ckpt(self, save_name, model, optimizer, epoch, step=None):
        if isinstance(model, torch.nn.DataParallel):
            model = model.mudule
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_name)
        if step is not None: 
            print('Save model: {} in epoch {} step {}'.format(save_name, epoch + 1, step))
        else:
            print('Save model: {} in epoch {} last step'.format(save_name, epoch + 1))

    def _clip_gradient(self, model, clip_norm):
        '''
        Computes a gradient clipping coefficient based on gradient norm.
        '''
        total_norm = 0
        for p in model.parameters():
            if p.requires_grad:
                module_norm = p.grad.data.norm()
                total_norm += module_norm ** 2
        total_norm = np.sqrt(total_norm)

        norm = clip_norm / max(total_norm, clip_norm)
        for p in model.parameters():
            if p.requires_grad:
                p.grad.mul_(norm)

    def _adjust_lr(self, epoch):
        if epoch % (self.pipe_cfg.TRAIN.LR_DECAY_EPOCH + 1) == 0 and epoch > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.pipe_cfg.TRAIN.LR_DECAY_GAMMA * param_group['lr']
                self.lr = self.pipe_cfg.TRAIN.LR_DECAY_GAMMA * self.lr

    def _prepare_model_input(self, data):
        '''
        Prepare model input from cpu data to device data, data is directly from data loader,
        it contains (inputs, targets), inputs or targets can be a tensor or a list/tuple of tensors
        '''
        inputs, targets = data
        if isinstance(inputs, (tuple, list)):
            inputs = [input_.to(self.device) for input_ in inputs]
        else:
            inputs = inputs.to(self.device)
        return inputs

    def _run_step_train(self, data):
        self.logger.tic()
        inputs = self._prepare_model_input(data)
        self.model.zero_grad()
        output_list = self.model(inputs)
        self.loss(data, output_list)
        loss = self.logger.total_loss()
        self.optimizer.zero_grad()
        # todo shouldn't it run after backward ? and crash
        if self.pipe_cfg.TRAIN.CLIP_GRAD != -1:
            self._clip_gradient(self.model, self.pipe_cfg.TRAIN.CLIP_GRAD)
        loss.backward()
        self.optimizer.step()
        self.logger.toc()

    def _run_epoch_train(self, epoch, num_steps_per_epoch, num_steps):
        self._adjust_lr(epoch)
        
        data_iter = iter(self.train_data_loader)
        for step in range(num_steps_per_epoch):
            data = next(data_iter)
            self._run_step_train(data)
            self.global_step += 1
            # Log if global step reach DISP_INTERVAL
            if self.global_step % self.pipe_cfg.TRAIN.DISP_INTERVAL == 0:
                self.logger.log_train(
                    epoch,
                    self.global_step,
                    self.lr,
                    num_steps,
                    step,
                    num_steps_per_epoch,
                    self.model.state_dict().items())

            # Save shapshot if every time that step reach SHAPSHOT_INTERVAL
            if step > 0 and step % self.pipe_cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                SNAPSHOT_DIR = os.path.join(self.pipe_cfg.TRAIN.RESULTS_ROOT, 'snapshots')
                save_name = os.path.join(SNAPSHOT_DIR, self.pipe_cfg.TRAIN.SNAPSHOT_PREFIX) + \
                            '_{}_{}.pth'.format(epoch + 1, step)
                self._save_ckpt(save_name, self.model, self.optimizer, epoch, step=step)

        # Save snapshot every epoch
        SNAPSHOT_DIR = os.path.join(self.pipe_cfg.TRAIN.RESULTS_ROOT, 'snapshots')
        save_name = os.path.join(SNAPSHOT_DIR, self.pipe_cfg.TRAIN.SNAPSHOT_PREFIX) + \
                    '_{}.pth'.format(epoch + 1)
        self._save_ckpt(save_name, self.model, self.optimizer, epoch)
        
    # todo set fix param and print fixed param
    def _run_train(self):
        start_epoch = self.pipe_cfg.TRAIN.START_EPOCH
        end_epoch = self.pipe_cfg.TRAIN.END_EPOCH
        batch_size = self.train_batch_size
        print('Epoch: start in {} end in {}'.format(start_epoch, end_epoch))

        num_epochs = end_epoch - 0 + 1  # Start from epoch 0
        num_steps_per_epoch = len(self.train_data_loader)
        num_steps = num_steps_per_epoch * num_epochs
        print('num_steps: {:d}  batch_size: {} num_steps_per_epoch: {}'.format(
            int(num_steps), batch_size, num_steps_per_epoch))

        self.global_step = start_epoch * num_steps_per_epoch
        self.lr = self.pipe_cfg.TRAIN.LEARNING_RATE

        start_train_time = time.time()

        self.model.train()  # set to train model, will affect bn stats and dropout
        for epoch in range(start_epoch, end_epoch):
            start = time.time()
            self._run_epoch_train(epoch, num_steps_per_epoch, num_steps) 
            end = time.time()
            print('Elapsed time for this epoch: {:.2f}s'.format(end - start))
            if self.pipe_cfg.TRAIN.EVAL_INTERVAL > 0 and \
                (epoch + 1) % self.pipe_cfg.TRAIN.EVAL_INTERVAL == 0:
                self._run_test()
                self.model.train()

        end_train_time = time.time()
        hours, remainder = divmod((end_train_time-start_train_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        print('Training cost: {:d}h{:d}m{:d}s'.format(int(hours), int(minutes), int(seconds)))
        print('\t end at: {}\n\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        self._run_deploy()

    def _run_test(self):
        EVAL_RESULTS_DIR = os.path.join(self.pipe_cfg.TRAIN.RESULTS_ROOT, 'eval')
        if not os.path.isdir(EVAL_RESULTS_DIR):
            os.makedirs(EVAL_RESULTS_DIR)       
        TEST_CACHE_FILE = os.path.join(EVAL_RESULTS_DIR, 'test_cache.pkl')

        batch_size = self.test_batch_size
        assert batch_size == 1, 'test batch must be 1.'
        num_data = len(self.test_data_loader)
        print('Test: data number total {}'.format(num_data))
        start_train_time = time.time()

        EVAL_USE_CACHE = self.pipe_cfg.TEST.EVAL_USE_CACHE
        if not os.path.exists(TEST_CACHE_FILE):
            print('No test cache file found, evaluate will use new test results.')
            EVAL_USE_CACHE = False

        if self.mode == 'train':
            print('In train mode, evaluate will use new test results.')
            EVAL_USE_CACHE = False

        self._eval_table.clear()

        if EVAL_USE_CACHE:
            print('Evaluate with test cache:')
            with open(TEST_CACHE_FILE, 'rb') as f:
                self._eval_table = pickle.load(f)
        else:
            self.model.eval()  # set to train model, will affect bn stats and dropout
            data_iter = iter(self.test_data_loader)
            for i in range(num_data):
                data_tic = time.time()
                data = next(data_iter)
                data_toc = time.time()

                pre_tic = time.time()
                inputs = self._prepare_model_input(data)
                pre_toc = time.time()

                net_tic = time.time()
                with torch.no_grad():
                    output_list = self.model(inputs)
                net_toc = time.time()
                post_tic = time.time()
                self.eval_step(data, output_list)
                post_toc = time.time()

                data_time = data_toc - data_tic
                pre_time = pre_toc - pre_tic
                net_time = net_toc - net_tic
                post_time = post_toc - post_tic
                sys.stdout.write(
                    'Test: {:d}/{:d} data:{:.3f}s pre:{:.3f}s net:{:.3f}s post:{:.3f}s\r'.format(
                        i + 1, num_data, data_time, pre_time, net_time, post_time))
                sys.stdout.flush()

        if self.pipe_cfg.TEST.WRITE_CACHE and self.mode != 'train':  
            with open(os.path.join(EVAL_RESULTS_DIR, 'test_cache.pkl'), 'wb') as f:
                pickle.dump(self._eval_table, f, pickle.HIGHEST_PROTOCOL)
        self.eval()
        self.logger.log_test(self.global_step, log_to_tensorboard=(self.model == 'train'))

        end_train_time = time.time()
        hours, remainder = divmod((end_train_time - start_train_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        print('Test cost: {:d}h{:d}m{:d}s'.format(int(hours), int(minutes), int(seconds)))
        print('\t end at: {}\n\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def _run_deploy(self):
        # Save onnx model
        onnx_file = os.path.join(self.pipe_cfg.TRAIN.RESULTS_ROOT, 
                                 '{}.onnx'.format(self.pipe_cfg.TRAIN.SNAPSHOT_PREFIX))
        data_iter = iter(self.test_data_loader)
        data = next(data_iter)
        inputs = self._prepare_model_input(data)
        torch.onnx.export(self.model, inputs, onnx_file, export_params=True)
        print('Onnx saved to {}, you can view the network structure at https://netron.app'
            .format(onnx_file))

        # Print onnx graph
        onnx_model = None
        try:
            import onnx 
            onnx_model = onnx.load(onnx_file)
            print("Onnx graph: \nop_type, input_names, output_names")
            for node in onnx_model.graph.node:
                print(node.op_type, node.input, node.output)
        except:
            print('WARNING: Failed to import onnx! can not print the graph node and can not '
                  'generate deploy model.json automatically, you can refer to '
                  'deploy/models/mnist/mnist.json to write it yourself, the input name and output '
                  'name can be get by https://netron.app')

        # Convert onnx model to deploy model
        if os.path.exists(self.pipe_cfg.DEPLOY.ONNX2NCNN):
            self._convert_deploy_model(onnx_file, inputs.shape[1:], onnx_model)
        else:
            print('Can not find onnx2ncnn tool, skip convert.'
                  'You can set the tool binary path to by PIPELINE.DEPLOY.ONNX2NCNN.')

    def _convert_deploy_model(self, onnx_file, input_shape, onnx_model=None):
        deploy_dir = os.path.join(self.pipe_cfg.TRAIN.RESULTS_ROOT, 'deploy')
        if not os.path.isdir(deploy_dir):
            os.makedirs(deploy_dir)

        ncnn_param_file = os.path.join(deploy_dir, 
                                      '{}.param'.format(self.pipe_cfg.TRAIN.SNAPSHOT_PREFIX))
        ncnn_bin_file = os.path.join(deploy_dir, 
                                     '{}.bin'.format(self.pipe_cfg.TRAIN.SNAPSHOT_PREFIX))
        cmd = '{} {} {} {}'.format(
            self.pipe_cfg.DEPLOY.ONNX2NCNN, onnx_file, ncnn_param_file, ncnn_bin_file)
        os.system(cmd)

        print('Convert onnx model to deploy(ncnn) model, deploy model saved to {} {}'.format(
              ncnn_param_file, ncnn_bin_file))

        if onnx_model is not None:
            deploy_json_file = os.path.join(deploy_dir,
                                            '{}.json'.format(self.pipe_cfg.TRAIN.SNAPSHOT_PREFIX))
            model_json = \
            [
                {
                    "type": "operator",
                    "name": "BackBone",
                    "inputs":
                    [
                        {
                            "name": onnx_model.graph.node[0].input[0],
                            "shape": [input_shape[1], input_shape[2], input_shape[0]]
                        }
                    ],
                    "outputs": [onnx_model.graph.node[-1].output[0]]
                },
                {
                    "type": "plugin",
                    "name": "PostProcess"
                }
            ]
            with open(deploy_json_file, 'w') as f:
                json.dump(model_json, f, indent=4)
            print('Generate deploy json file, saved to {}'.format(deploy_json_file))

def _parse_args():
    parser = argparse.ArgumentParser(description='Deeplearning train test deploy pipeline')
    parser.add_argument('--cfg', help='config file', type=str)
    parser.add_argument('--mode', help='train test or deploy mode', type=str)
    parser.add_argument('--cuda', help='whether use CUDA', action='store_true')
    args = parser.parse_args()
    return args

def _add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

args = _parse_args()

if __name__ == '__main__':
    _add_path(os.path.dirname(__file__))
    Pipeline(cfg_file=args.cfg, mode=args.mode, use_cuda=args.cuda)

    # Test code
    # Pipeline(cfg_file='yml.cfg', mode='train', use_cuda=False)
