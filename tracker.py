# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import random
import torch
from copy import deepcopy
from dbquery import DBQuery
from datamanager import expand_da

class StateTracker(object):
    def __init__(self, data_dir, config):
        self.time_step = 0
        self.cfg = config
        self.db = DBQuery(data_dir)
        self.topic = 'NONE'
    
    def _action_to_dict(self, das):
        da_dict = {}
        for da, value in das.items():
            domain, intent, slot, p = da.split('-')
            domint = '-'.join((domain, intent))
            if domint not in da_dict:
                da_dict[domint] = []
            da_dict[domint].append([slot, value])
        return da_dict
    
    def _dict_to_vec(self, das):
        da_vector = torch.zeros(self.cfg.a_dim_usr, dtype=torch.int32)
        expand_da(das)
        for domint in das:
            pairs = das[domint]
            for slot, p, value in pairs:
                da = '-'.join((domint, slot, p)).lower()
                if da in self.cfg.dau2idx:
                    idx = self.cfg.dau2idx[da]
                    da_vector[idx] = 1
        return da_vector
    
    def _mask_user_goal(self, goal):
        domain_ordering = list(goal['domain_ordering'])
        if 'hospital' in goal:
            del(goal['hospital'])
            domain_ordering.remove('hospital')
        if 'police' in goal:
            del(goal['police'])
            domain_ordering.remove('police')
        goal['domain_ordering'] = tuple(domain_ordering)
    
    def get_entities(self, s, domain):
        origin = s['belief_state']['inform'][domain].items()
        constraint = []
        for k, v in origin:
            if k in self.cfg.mapping[domain]:
                constraint.append((self.cfg.mapping[domain][k], v))
        entities = self.db.query(domain, constraint)
        random.shuffle(entities)
        return entities
        
    def update_belief_sys(self, old_s, a):
        """
        update belief state with sys action
        """
        s = deepcopy(old_s)
        a_index = torch.nonzero(a) # get multiple da indices
        
        self.time_step += 1
        s['others']['turn'] = self.time_step
            
        # update sys/user dialog act
        s['history']['sys'] = dict(s['history']['sys'], **s['last_sys_action'])
        del(s['last_sys_action'])
        s['last_user_action'] = s['user_action']
        s['user_action'] = dict()
        
        # update belief part
        das = [self.cfg.idx2da[idx.item()] for idx in a_index]
        das = [da.split('-') for da in das]
        sorted(das, key=lambda x:x[0]) # sort by domain
        
        entities = [] if self.topic == 'NONE' else self.get_entities(s, self.topic)
        for domain, intent, slot, p in das:
            _domain = self.topic if domain == 'booking' else domain
            if domain in self.cfg.belief_domains and domain != self.topic:
                self.topic = domain
                entities = self.get_entities(s, domain)
                
            da = '-'.join((domain, intent, slot, p))
            if intent in ['nooffer', 'nobook']:
                if slot in s['belief_state']['inform'][_domain]:
                    s['sys_action'][da] = s['belief_state']['inform'][_domain][slot]
                else:
                    s['sys_action'][da] = 'none'
            elif slot == 'choice':
                s['sys_action'][da] = str(len(entities))
            elif p == 'none':
                s['sys_action'][da] = 'none'
            elif p == '?':
                s['sys_action'][da] = '?'
            else:
                num = int(p) - 1
                if len(entities) > num and slot in self.cfg.mapping[_domain]:
                    typ = self.cfg.mapping[_domain][slot]
                    s['sys_action'][da] = entities[num][typ]
                else:
                    s['sys_action'][da] = 'none'
            
            if intent == 'inform' and _domain != 'NONE':
                s['belief_state']['request'][_domain].discard(slot)
            
            # booked
            if intent == 'inform' and slot == 'car': # taxi
                if not s['belief_state']['booked']['taxi']:
                    s['belief_state']['booked']['taxi'] == 'booked'
            elif intent == 'offerbooked' and slot == 'ref': # train
                s['belief_state']['request']['train'].discard('ref')
                if not s['belief_state']['booked']['train'] and entities:
                    s['belief_state']['booked']['train'] = entities[0]['ref']
            elif intent == 'book' and slot == 'ref': # attraction, hotel, restaurant
                if _domain not in ['attraction', 'hotel', 'restaurant']:
                    continue
                s['belief_state']['request'][_domain].discard('ref')
                if not s['belief_state']['booked'][_domain] and entities:
                    # save entity id
                    s['belief_state']['booked'][_domain] = entities[0]['ref']
        
        return s
    
    def update_belief_usr(self, old_s, a, terminal):
        """
        update belief state with user action
        """
        s = deepcopy(old_s)
        a_index = torch.nonzero(a) # get multiple da indices
        
        self.time_step += 1
        s['others']['turn'] = self.time_step
        s['others']['terminal'] = terminal
        
        # update sys/user dialog act
        s['history']['user'] = dict(s['history']['user'], **s['last_user_action'])
        del(s['last_user_action'])
        s['last_sys_action'] = s['sys_action']
        s['sys_action'] = dict()
        
        # update belief part
        das = [self.cfg.idx2dau[idx.item()] for idx in a_index]
        das = [da.split('-') for da in das]
        sorted(das, key=lambda x:x[0]) # sort by domain
        
        for domain, intent, slot, p in das:
            if domain in self.cfg.belief_domains and domain != self.topic:
                self.topic = domain
            
            da = '-'.join((domain, intent, slot, p))
            if p == 'none':
                s['user_action'][da] = 'none'
            elif p == '?':
                s['user_action'][da] = '?'
            else:
                if slot in s['user_goal']['inform'][domain]:
                    s['user_action'][da] = s['user_goal']['inform'][domain][slot]
                else:
                    s['user_action'][da] = 'dont care'
            
            if slot != 'none':
                if intent == 'inform':
                    # update constraints with reasonable value according to user goal
                    if slot in s['user_goal']['inform'][domain]:
                        s['belief_state']['inform'][domain][slot] = s['user_goal']['inform'][domain][slot] # value
                    else:
                        s['belief_state']['inform'][domain][slot] = 'dont care'
                
                elif intent == 'request':
                    s['belief_state']['request'][domain].add(slot)
        
        return s
    
    def reset(self, random_seed=None):
        """
        Args:
            random_seed (int):
        Returns:
            init_state (dict):
        """
        pass
    
    def step(self, s, sys_a):
        """
        Args:
            s (dict):
            sys_a (vector):
        Returns:
            next_s (dict):
            terminal (bool):
        """
        pass
