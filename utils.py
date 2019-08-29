# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import time
import logging
import os
import numpy as np
import argparse
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--log_dir', type=str, default='log', help='Logging directory')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='model', help='Directory to store model')
    parser.add_argument('--load', type=str, default='', help='File name to load trained model')
    parser.add_argument('--load_user', type=str, default='', help='File name to load user simulator')
    parser.add_argument('--pretrain', type=bool, default=False, help='Set to pretrain')
    parser.add_argument('--test', type=bool, default=False, help='Set to inference')
    parser.add_argument('--config', type=str, default='multiwoz', help='Dataset to use')
    parser.add_argument('--simulator', type=str, default='agenda', help='User simulator to use')
    
    parser.add_argument('--epoch', type=int, default=32, help='Max number of epoch')
    parser.add_argument('--save_per_epoch', type=int, default=5, help="Save model every XXX epoches")
    parser.add_argument('--process', type=int, default=16, help='Process number')
    parser.add_argument('--batchsz', type=int, default=32, help='Batch size')
    parser.add_argument('--batchsz_traj', type=int, default=1024, help='Batch size to collect trajectories')
    parser.add_argument('--print_per_batch', type=int, default=400, help="Print log every XXX batches")
    parser.add_argument('--update_round', type=int, default=5, help='Epoch num for inner loop of PPO')
    parser.add_argument('--lr_rl', type=float, default=3e-4, help='Learning rate of dialog policy')
    parser.add_argument('--lr_irl', type=float, default=1e-3, help='Learning rate of reward estimator')
    parser.add_argument('--lr_simu', type=float, default=1e-3, help='Learning rate of user simulator')
    
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted factor')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Clip epsilon of ratio r(theta)')
    parser.add_argument('--tau', type=float, default=0.95, help='Generalized advantage estimation')
    parser.add_argument('--anneal', type=int, default=5000, help='Max steps for annealing')
    parser.add_argument('--clip', type=float, default=0.02, help='Clipping parameter on WGAN')

    return parser

def init_session(key, cfg):
    turn_data = {}
    turn_data['others'] = {'session_id':key, 'turn':0, 'terminal':False}
    turn_data['sys_action'] = dict()
    turn_data['user_action'] = dict()
    turn_data['history'] = {'sys':dict(), 'user':dict()}
    turn_data['belief_state'] = {'inform':{}, 'request':{}, 'booked':{}}
    for domain in cfg.belief_domains:
        turn_data['belief_state']['inform'][domain] = dict()
        turn_data['belief_state']['request'][domain] = set()
        turn_data['belief_state']['booked'][domain] = ''
    
    session_data = {'inform':{}, 'request':{}, 'book':{}}
    for domain in cfg.belief_domains:
        session_data['inform'][domain] = dict()
        session_data['request'][domain] = set()
        session_data['book'][domain] = False
    
    return turn_data, session_data

def init_goal(dic, goal, cfg):
    for domain in cfg.belief_domains:
        if domain in goal and goal[domain]:
            domain_data = goal[domain]
            # constraint
            if 'info' in domain_data and domain_data['info']:
                for slot, value in domain_data['info'].items():
                    slot = cfg.map_inverse[domain][slot]
                    # single slot value for user goal
                    inform_da = domain+'-'+slot+'-1'
                    if inform_da in cfg.inform_da:
                        dic['inform'][domain][slot] = value
            # booking
            if 'book' in domain_data and domain_data['book']:
                dic['book'][domain] = True
            # request
            if 'reqt' in domain_data and domain_data['reqt']:
                for slot in domain_data['reqt']:
                    slot = cfg.map_inverse[domain][slot]
                    request_da = domain+'-'+slot
                    if request_da in cfg.request_da:
                        dic['request'][domain].add(slot)

def init_logging_handler(log_dir, extra=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(log_dir, current_time+extra))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

def to_device(data):
    if type(data) == dict:
        for k, v in data.items():
            data[k] = v.to(device=DEVICE)
    else:
        for idx, item in enumerate(data):
            data[idx] = item.to(device=DEVICE)
    return data

def state_vectorize(state, config, db, noisy=False):
    """
    state: dict_keys(['user_action', 'sys_action', 'user_goal', 'belief_state', 'history', 'others']) 
    state_vec: [user_act, last_sys_act, inform, request, book, degree]
    """
    user_act = np.zeros(len(config.da_usr))
    for da in state['user_action']:
        user_act[config.dau2idx[da]] = 1.
    
    last_sys_act = np.zeros(len(config.da))
    for da in state['last_sys_action']:
        last_sys_act[config.da2idx[da]] = 1.
    
    user_history = np.zeros(len(config.da_usr))
    for da in state['history']['user']:
        user_history[config.dau2idx[da]] = 1.
    
    sys_history = np.zeros(len(config.da))
    for da in state['history']['sys']:
        sys_history[config.da2idx[da]] = 1.
    
    inform = np.zeros(len(config.inform_da))
    for domain in state['belief_state']['inform']:
        for slot, value in state['belief_state']['inform'][domain].items():
            dom_slot, p = domain+'-'+slot+'-', 1
            key = dom_slot + str(p)
            while inform[config.inform2idx[key]]:
                p += 1
                key = dom_slot + str(p)
                if key not in config.inform2idx:
                    break
            else:
                inform[config.inform2idx[key]] = 1.
            
    request = np.zeros(len(config.request_da))
    for domain in state['belief_state']['request']:
        for slot in state['belief_state']['request'][domain]:
            request[config.request2idx[domain+'-'+slot]] = 1.
    
    book = np.zeros(len(config.belief_domains))
    for domain in state['belief_state']['booked']:
        if state['belief_state']['booked'][domain]:
            book[config.domain2idx[domain]] = 1.
    
    degree = db.pointer(state['belief_state']['inform'], config.mapping, config.db_domains, noisy)
        
    final = 1. if state['others']['terminal'] else 0.
    
    state_vec = np.r_[user_act, last_sys_act, user_history, sys_history, inform, request, book, degree, final]
    assert len(state_vec) == config.s_dim
    return state_vec

def action_vectorize(action, config):
    act_vec = np.zeros(config.a_dim)
    for da in action['sys_action']:
        act_vec[config.da2idx[da]] = 1
    return act_vec

def reparameterize(mu, logvar):
    std = (0.5*logvar).exp()
    eps = torch.randn_like(std)
    return eps.mul(std) + mu
