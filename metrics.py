# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import numpy as np
from dbquery import DBQuery

class Evaluator(object):
    def __init__(self, data_dir, cfg):
        self.db = DBQuery(data_dir)
        self.cfg = cfg

    def _init_dict(self):
        dic = {}
        for domain in self.cfg.belief_domains:
            dic[domain] = {}
        return dic

    def match_rate_goal(self, goal, booked_entity):
        """
        judge if the selected entity meets the constraint
        """
        score = []
        for domain in self.cfg.belief_domains:
            if goal['book'][domain]:
                tot = len(goal['inform'][domain].keys())
                if tot == 0:
                    continue
                entity_id = booked_entity[domain]
                if not entity_id:
                    score.append(0)
                    continue
                if domain == 'taxi':
                    score.append(1)
                    continue
                match = 0
                entity = self.db.dbs[domain][int(entity_id)]
                for k, v in goal['inform'][domain].items():
                    k = self.cfg.mapping[domain][k]
                    if k == 'leaveAt':
                        try:
                            v_constraint = int(v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['leaveAt'].split(':')[0]) * 100 + int(entity['leaveAt'].split(':')[1])
                            if v_constraint <= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    elif k == 'arriveBy':
                        try:
                            v_constraint = int(v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['arriveBy'].split(':')[0]) * 100 + int(entity['arriveBy'].split(':')[1])
                            if v_constraint >= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    else:
                        if v.strip() == entity[k].strip():
                            match += 1
                score.append(match / tot)
        return score
    
    def inform_F1_goal(self, goal, sys_history):
        """
        judge if all the requested information is answered
        """
        inform_slot = {}
        for domain in self.cfg.belief_domains:
            inform_slot[domain] = set()
        for da in sys_history:
            domain, intent, slot, p = da.split('-')
            if intent in ['inform', 'recommend', 'offerbook', 'offerbooked'] and slot != 'none' and domain in self.cfg.belief_domains:
                inform_slot[domain].add(slot)
        TP, FP, FN = 0, 0, 0
        for domain in self.cfg.belief_domains:
            for k in goal['request'][domain]:
                if k in inform_slot[domain]:
                    TP += 1
                else:
                    FN += 1
            for k in inform_slot[domain]:
                # exclude slots that are informed by users
                if k not in goal['request'][domain] and k not in goal['inform'][domain] and k in self.cfg.requestable[domain]:
                    FP += 1
        return TP, FP, FN
    
    def match_rate(self, metadata, aggregate=False):
        booked_entity = metadata['belief_state']['booked']
        """
        goal = {'book':{}, 'inform':self._init_dict()}
        goal['book'] = metadata['user_goal']['book']
        for da, v in metadata['history']['user'].items():
            d, i, s, p = da.split('-')
            if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in self.cfg.mapping[d]:
                goal['inform'][d][s] = v
        """
        goal = metadata['user_goal']
        score = self.match_rate_goal(goal, booked_entity)
        if not aggregate:
            return score
        else:
            return np.mean(score) if score else None

    def inform_F1(self, metadata, aggregate=False):
        sys_history = dict(metadata['history']['sys'], **metadata['last_sys_action'])
        """
        goal = {'request':self._init_dict(), 'inform':self._init_dict()}
        for da, v in metadata['history']['user'].items():
            d, i, s, p = da.split('-')
            if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in self.cfg.mapping[d]:
                goal['inform'][d][s] = v
            elif i == 'request':
                goal['request'][d][s] = v
        """
        goal = metadata['user_goal']
        TP, FP, FN = self.inform_F1_goal(goal, sys_history)
        if not aggregate:
            return [TP, FP, FN]
        else:    
            try:
                rec = TP / (TP + FN)
            except ZeroDivisionError:
                return None, None, None
            try:
                prec = TP / (TP + FP)
                F1 = 2 * prec * rec / (prec + rec)
            except ZeroDivisionError:
                return 0, rec, 0
            return prec, rec, F1
