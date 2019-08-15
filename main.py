# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import sys
import time
import logging
from utils import get_parser, init_logging_handler
from datamanager import DataManager
from user import UserNeural
from usermanager import UserDataManager
from agenda import UserAgenda
from ppo import PPO
from config import MultiWozConfig
from torch import multiprocessing as mp

def worker_user(args, manager, config):
    init_logging_handler(args.log_dir, '_user')
    env = UserNeural(args, manager, config, True)
    
    best = float('inf')
    for e in range(args.epoch):
        env.imitating(e)
        best = env.imit_test(e, best)

def worker_policy(args, manager, config):
    init_logging_handler(args.log_dir, '_policy')
    agent = PPO(None, args, manager, config, 0, pre=True)
    
    best = float('inf')
    for e in range(args.epoch):
        agent.imitating(e)
        best = agent.imit_test(e, best)

def worker_estimator(args, manager, config, make_env):
    init_logging_handler(args.log_dir, '_estimator')
    agent = PPO(make_env, args, manager, config, args.process, pre_irl=True)
    agent.load(args.save_dir+'/best')
    
    best0, best1 = float('inf'), float('inf')
    for e in range(args.epoch):
        agent.train_irl(e, args.batchsz_traj)
        best0 = agent.test_irl(e, args.batchsz, best0)
        best1 = agent.imit_value(e, args.batchsz_traj, best1)

def make_env_neural():
    env = UserNeural(args, usermanager, config)
    env.load(args.load_user)
    return env

def make_env_agenda():
    env = UserAgenda(args.data_dir, config)
    return env

if __name__ == '__main__':
    parser = get_parser()
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)
    if not args.load_user:
        args.load_user = args.save_dir+'/best'
    
    if args.config == 'multiwoz':
        config = MultiWozConfig()
    else:
        raise NotImplementedError('Config of the dataset {} not implemented'.format(args.config))
    if args.simulator == 'neural':
        usermanager = UserDataManager(args.data_dir, config.data_file)
        make_env = make_env_neural
    elif args.simulator == 'agenda':
        make_env = make_env_agenda
    else:
        raise NotImplementedError('User simulator {} not implemented'.format(args.simulator))
    init_logging_handler(args.log_dir)
    logging.debug(str(args))
    
    manager = DataManager(args.data_dir, config)
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    if args.pretrain:
        logging.debug('pretrain')
        
        processes = []
        process_args = (args, manager, config)
        processes.append(mp.Process(target=worker_policy, args=process_args))
        if args.simulator == 'neural':
            process_args_user = (args, usermanager, config)
            processes.append(mp.Process(target=worker_user, args=process_args_user))
        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
        worker_estimator(args, manager, config, make_env)
            
    elif args.test:
        logging.debug('test')
        logging.disable(logging.DEBUG)
    
        agent = PPO(make_env, args, manager, config, 1, infer=True)
        agent.load(args.load)
        agent.evaluate()
        
    else: # training
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        logging.debug('train {}'.format(current_time))
    
        agent = PPO(make_env, args, manager, config, args.process)
        best = agent.load(args.load)
        
        # auto, irl, rl
        for i in range(args.epoch):
            agent.update(args.batchsz_traj, i)
            # validation
            best = agent.update(args.batchsz, i, best)
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.debug('epoch {} {}'.format(i, current_time))
