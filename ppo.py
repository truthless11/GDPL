# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import os
import numpy as np
import logging
import pickle

import torch
import torch.nn as nn
from torch import optim
from torch import multiprocessing as mp

from rlmodule import MultiDiscretePolicy, Value, Memory, Transition
from estimator import RewardEstimator
from utils import state_vectorize, to_device 
from metrics import Evaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sampler(pid, queue, evt, env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 40
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(state_vectorize(s, env.cfg, env.db))
            a = policy.select_action(s_vec.to(device=DEVICE)).cpu()

            # interact with env
            next_s, done = env.step(s, a)

            # a flag indicates ending or not
            mask = 0 if done else 1
            
            # get reward compared to demostrations
            next_s_vec = torch.Tensor(state_vectorize(next_s, env.cfg, env.db))
            
            # save to queue
            buff.push(s_vec.numpy(), a.numpy(), mask, next_s_vec.numpy())

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


class PPO(object):
    def __init__(self, env_cls, args, manager, cfg, process_num, pre=False, pre_irl=False, infer=False):
        """
        :param env_cls: env class or function, not instance, as we need to create several instance in class.
        :param args:
        :param manager:
        :param cfg:
        :param process_num: process number
        :param pre: set to pretrain mode
        :param infer: set to test mode
        """

        self.process_num = process_num

        # initialize envs for each process
        self.env_list = []
        for _ in range(process_num):
            self.env_list.append(env_cls())

        # construct policy and value network
        self.policy = MultiDiscretePolicy(cfg).to(device=DEVICE)
        self.value = Value(cfg).to(device=DEVICE)
        
        if pre:
            self.print_per_batch = args.print_per_batch
            from dbquery import DBQuery
            db = DBQuery(args.data_dir)
            self.data_train = manager.create_dataset_rl('train', args.batchsz, cfg, db)
            self.data_valid = manager.create_dataset_rl('valid', args.batchsz, cfg, db)
            self.data_test = manager.create_dataset_rl('test', args.batchsz, cfg, db)
            self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()
        else:
            self.rewarder = RewardEstimator(args, manager, cfg, pretrain=pre_irl, inference=infer)
            self.evaluator = Evaluator(args.data_dir, cfg)

        self.save_dir = args.save_dir
        self.save_per_epoch = args.save_per_epoch
        self.optim_batchsz = args.batchsz
        self.update_round = args.update_round
        self.policy.eval()
        self.value.eval()
        
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.tau = args.tau
        self.policy_optim = optim.RMSprop(self.policy.parameters(), lr=args.lr_rl)
        self.value_optim = optim.Adam(self.value.parameters(), lr=args.lr_rl)

    def policy_loop(self, data):
        s, target_a = to_device(data)
        a_weights = self.policy(s)
        
        loss_a = self.multi_entropy_loss(a_weights, target_a)
        return loss_a

    def imitating(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            loss_a.backward()
            self.policy_optim.step()
            
            if (i+1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_a:{}'.format(epoch, i, a_loss))
                a_loss = 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch, True)
        self.policy.eval()
    
    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """        
        a_loss = 0.
        for i, data in enumerate(self.data_valid):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            
        a_loss /= len(self.data_valid)
        logging.debug('<<dialog policy>> validation, epoch {}, loss_a:{}'.format(epoch, a_loss))
        if a_loss < best:
            logging.info('<<dialog policy>> best model saved')
            best = a_loss
            self.save(self.save_dir, 'best', True)
            
        a_loss = 0.
        for i, data in enumerate(self.data_test):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            
        a_loss /= len(self.data_test)
        logging.debug('<<dialog policy>> test, epoch {}, loss_a:{}'.format(epoch, a_loss))
        return best
    
    def imit_value(self, epoch, batchsz, best):
        self.value.train()
        batch = self.sample(batchsz)
        s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
        batchsz = s.size(0)
        
        v = self.value(s).squeeze(-1).detach()
        log_pi_old_sa = self.policy.get_log_prob(s, a).detach()
        r = self.rewarder.estimate(s, a, next_s, log_pi_old_sa).detach()
        A_sa, v_target = self.est_adv(r, v, mask)
        
        for i in range(self.update_round):
            perm = torch.randperm(batchsz)
            v_target_shuf, s_shuf = v_target[perm], s[perm]
            optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            v_target_shuf, s_shuf = torch.chunk(v_target_shuf, optim_chunk_num), torch.chunk(s_shuf, optim_chunk_num)
            
            value_loss = 0.
            for v_target_b, s_b in zip(v_target_shuf, s_shuf):
                self.value_optim.zero_grad()
                v_b = self.value(s_b).squeeze(-1)
                loss = (v_b - v_target_b).pow(2).mean()
                value_loss += loss.item()
                loss.backward()
                self.value_optim.step()
        
            value_loss /= optim_chunk_num
            logging.debug('<<dialog policy>> epoch {}, iteration {}, loss {}'.format(epoch, i, value_loss))
            
        if value_loss < best:
            logging.info('<<dialog policy>> best model saved')
            best = value_loss
            self.save(self.save_dir, 'best', True)
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch, True)
        self.value.eval()
        return best
    
    def train_irl(self, epoch, batchsz):
        batch = self.sample(batchsz)
        self.rewarder.train_irl(batch, epoch)
    
    def test_irl(self, epoch, batchsz, best):
        batch = self.sample(batchsz)
        best = self.rewarder.test_irl(batch, epoch, best)
        return best

    def est_adv(self, r, v, mask):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param v: estimated value, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :return: A(s, a), V-target(s), both Tensor
        """
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        v_target = torch.Tensor(batchsz).to(device=DEVICE)
        delta = torch.Tensor(batchsz).to(device=DEVICE)
        A_sa = torch.Tensor(batchsz).to(device=DEVICE)

        prev_v_target = 0
        prev_v = 0
        prev_A_sa = 0
        for t in reversed(range(batchsz)):
            # mask here indicates a end of trajectory
            # this value will be treated as the target value of value network.
            # mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
            # formula: V(s_t) = r_t + gamma * V(s_t+1)
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

            # please refer to : https://arxiv.org/abs/1506.02438
            # for generalized adavantage estimation
            # formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

            # formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
            # here use symbol tau as lambda, but original paper uses symbol lambda.
            A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]

            # update previous
            prev_v_target = v_target[t]
            prev_v = v[t]
            prev_A_sa = A_sa[t]

        # normalize A_sa
        A_sa = (A_sa - A_sa.mean()) / A_sa.std()

        return A_sa, v_target

    def update(self, batchsz, epoch, best=None):
        """
        firstly sample batchsz items and then perform optimize algorithms.
        :param batchsz:
        :param epoch:
        :param best:
        :return:
        """
        backward = True if best is None else False
        if backward:
            self.policy.train()
            self.value.train()
        
        # 1. sample data asynchronously
        batch = self.sample(batchsz)

        # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
        # batch.action: ([1, a_dim], [1, a_dim]...)
        # batch.reward/ batch.mask: ([1], [1]...)
        s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
        batchsz = s.size(0)
        
        # 2. update reward estimator
        inputs = (s, a, next_s)
        if backward:
            self.rewarder.update_irl(inputs, batchsz, epoch)
        else:
            best[1] = self.rewarder.update_irl(inputs, batchsz, epoch, best[1])
            
        # 3. get estimated V(s) and PI_old(s, a)
        # actually, PI_old(s, a) can be saved when interacting with env, so as to save the time of one forward elapsed
        # v: [b, 1] => [b]
        v = self.value(s).squeeze(-1).detach()
        log_pi_old_sa = self.policy.get_log_prob(s, a).detach()
        
        # 4. estimate advantage and v_target according to GAE and Bellman Equation
        r = self.rewarder.estimate(s, a, next_s, log_pi_old_sa).detach()
        A_sa, v_target = self.est_adv(r, v, mask)
        if backward:
            logging.debug('<<dialog policy>> epoch {}, reward {}'.format(epoch, r.mean().item()))
        else:
            reward = r.mean().item()
            logging.debug('<<dialog policy>> validation, epoch {}, reward {}'.format(epoch, reward))
            if reward > best[2]:
                logging.info('<<dialog policy>> best model saved')
                best[2] = reward
                self.save(self.save_dir, 'best', True)
            with open(self.save_dir+'/best.pkl', 'wb') as f:
                pickle.dump(best, f)
            return best
        
        # 5. update dialog policy
        for i in range(self.update_round):

            # 1. shuffle current batch
            perm = torch.randperm(batchsz)
            # shuffle the variable for mutliple optimize
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = v_target[perm], A_sa[perm], s[perm], a[perm], \
                                                                           log_pi_old_sa[perm]

            # 2. get mini-batch for optimizing
            optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            # chunk the optim_batch for total batch
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
                                                                           torch.chunk(A_sa_shuf, optim_chunk_num), \
                                                                           torch.chunk(s_shuf, optim_chunk_num), \
                                                                           torch.chunk(a_shuf, optim_chunk_num), \
                                                                           torch.chunk(log_pi_old_sa_shuf,
                                                                                       optim_chunk_num)
            # 3. iterate all mini-batch to optimize
            policy_loss, value_loss = 0., 0.
            for v_target_b, A_sa_b, s_b, a_b, log_pi_old_sa_b in zip(v_target_shuf, A_sa_shuf, s_shuf, a_shuf,
                                                                     log_pi_old_sa_shuf):
                # print('optim:', batchsz, v_target_b.size(), A_sa_b.size(), s_b.size(), a_b.size(), log_pi_old_sa_b.size())
                # 1. update value network
                self.value_optim.zero_grad()
                v_b = self.value(s_b).squeeze(-1)
                loss = (v_b - v_target_b).pow(2).mean()
                value_loss += loss.item()
                
                # backprop
                loss.backward()
                # nn.utils.clip_grad_norm(self.value.parameters(), 4)
                self.value_optim.step()

                # 2. update policy network by clipping
                self.policy_optim.zero_grad()
                # [b, 1]
                log_pi_sa = self.policy.get_log_prob(s_b, a_b)
                # ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
                # we use log_pi for stability of numerical operation
                # [b, 1] => [b]
                ratio = (log_pi_sa - log_pi_old_sa_b).exp().squeeze(-1)
                surrogate1 = ratio * A_sa_b
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa_b
                # this is element-wise comparing.
                # we add negative symbol to convert gradient ascent to gradient descent
                surrogate = - torch.min(surrogate1, surrogate2).mean()
                policy_loss += surrogate.item()

                # backprop
                surrogate.backward()
                # gradient clipping, for stability
                torch.nn.utils.clip_grad_norm(self.policy.parameters(), 10)
                # self.lock.acquire() # retain lock to update weights
                self.policy_optim.step()
                # self.lock.release() # release lock
            
            value_loss /= optim_chunk_num
            policy_loss /= optim_chunk_num
            logging.debug('<<dialog policy>> epoch {}, iteration {}, value, loss {}'.format(epoch, i, value_loss))
            logging.debug('<<dialog policy>> epoch {}, iteration {}, policy, loss {}'.format(epoch, i, policy_loss))
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
            with open(self.save_dir+'/'+str(epoch)+'.pkl', 'wb') as f:
                pickle.dump(best, f)
        self.policy.eval()
        self.value.eval()

    def sample(self, batchsz):
        """
        Given batchsz number of task, the batchsz will be splited equally to each processes
        and when processes return, it merge all data and return
        :param batchsz:
        :return: batch
        """

        # batchsz will be splitted into each process,
        # final batchsz maybe larger than batchsz parameters
        process_batchsz = np.ceil(batchsz / self.process_num).astype(np.int32)
        # buffer to save all data
        queue = mp.Queue()

        # start processes for pid in range(1, processnum)
        # if processnum = 1, this part will be ignored.
        # when save tensor in Queue, the process should keep alive till Queue.get(),
        # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
        # however still some problem on CUDA tensors on multiprocessing queue,
        # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
        # so just transform tensors into numpy, then put them into queue.
        evt = mp.Event()
        processes = []
        for i in range(self.process_num):
            process_args = (i, queue, evt, self.env_list[i], self.policy, process_batchsz)
            processes.append(mp.Process(target=sampler, args=process_args))
        for p in processes:
            # set the process as daemon, and it will be killed once the main process is stoped.
            p.daemon = True
            p.start()

        # we need to get the first Memory object and then merge others Memory use its append function.
        pid0, buff0 = queue.get()
        for _ in range(1, self.process_num):
            pid, buff_ = queue.get()
            buff0.append(buff_) # merge current Memory into buff0
        evt.set()

        # now buff saves all the sampled data
        buff = buff0

        return buff.get_batch()
    
    def evaluate(self):
        env = self.env_list[0]
        traj_len = 40
        reward_tot, turn_tot, inform_tot, match_tot, success_tot = [], [], [], [], []
        for seed in range(1000):
            s = env.reset(seed)
            print('seed', seed)
            print('goal', env.goal.domain_goals)
            print('usr', s['user_action'])
            turn = traj_len
            reward = []
            value = []
            mask = []
            for t in range(traj_len):
                s_vec = torch.Tensor(state_vectorize(s, env.cfg, env.db)).to(device=DEVICE)
                # mode with policy during evaluation
                a = self.policy.select_action(s_vec, False)
                next_s, done = env.step(s, a.cpu())
                next_s_vec = torch.Tensor(state_vectorize(next_s, env.cfg, env.db)).to(device=DEVICE)
                log_pi = self.policy.get_log_prob(s_vec, a)
                r = self.rewarder.estimate(s_vec, a, next_s_vec, log_pi)
                v = self.value(s_vec).squeeze(-1)
                reward.append(r.item())
                value.append(v.item())
                s = next_s
                print('sys', s['last_sys_action'])
                print('usr', s['user_action'])
                if done:
                    mask.append(0)
                    turn = t+2 # one due to counting from 0, the one for the last turn
                    break
                mask.append(1)

            reward_tot.append(np.mean(reward))
            turn_tot.append(turn)
            match_tot += self.evaluator.match_rate(s)
            inform_tot.append(self.evaluator.inform_F1(s))
            reward = torch.Tensor(reward)
            value = torch.Tensor(value)
            mask = torch.LongTensor(mask)
            A_sa, v_target = self.est_adv(reward, value, mask)
            print('turn', turn)
            #print('reward', A_sa.tolist())
            print('reward', v_target[0].item())
            match_session = self.evaluator.match_rate(s, False)
            print('match', match_session)
            inform_session = self.evaluator.inform_F1(s, False)
            print('inform', inform_session)
            if (match_session == 1 and inform_session[1] == 1) \
            or (match_session == 1 and inform_session[1] is None) \
            or (match_session is None and inform_session[1] == 1):
                print('success', 1)
                success_tot.append(1)
            else:
                print('success', 0)
                success_tot.append(0)
        
        logging.info('reward {}'.format(np.mean(reward_tot)))
        logging.info('turn {}'.format(np.mean(turn_tot)))
        logging.info('match {}'.format(np.mean(match_tot)))
        TP, FP, FN = np.sum(inform_tot, 0)
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        F1 = 2 * prec * rec / (prec + rec)
        logging.info('inform rec {}, F1 {}'.format(rec, F1))
        logging.info('success {}'.format(np.mean(success_tot)))

    def save(self, directory, epoch, rl_only=False):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        if not rl_only:
            self.rewarder.save_irl(directory, epoch)

        torch.save(self.value.state_dict(), directory + '/' + str(epoch) + '_ppo.val.mdl')
        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_ppo.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    def load(self, filename):
        self.rewarder.load_irl(filename)
        
        value_mdl = filename + '_ppo.val.mdl'
        policy_mdl = filename + '_ppo.pol.mdl'
        if os.path.exists(value_mdl):
            self.value.load_state_dict(torch.load(value_mdl))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(value_mdl))
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
        
        best_pkl = filename + '.pkl'
        if os.path.exists(best_pkl):
            with open(best_pkl, 'rb') as f:
                best = pickle.load(f)
        else:
            best = [float('inf'),float('inf'),float('-inf')]
        return best
