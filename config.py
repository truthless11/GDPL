# -*- coding: utf-8 -*-
"""
@author: truthless
"""

class Config():
    
    def __init__(self):
        self.domain = []
        self.intent = []
        self.slot = []
        self.da = []
        self.da_usr = []
        self.data_file = ''
        self.db_domains = []
        self.belief_domains = []
        
    def init_inform_request(self):
        self.inform_da = []
        self.request_da = []
        self.requestable = {}
        for domain in self.belief_domains:
            self.requestable[domain] = []
        
        for da in self.da_usr:
            d, i, s, v = da.split('-')
            if s == 'none':
                continue
            if i == 'inform':
                self.inform_da.append('-'.join([d,s,v]))
            elif i == 'request':
                self.request_da.append('-'.join([d,s]))
                self.requestable[d].append(s)
        
    def init_dict(self):
        self.domain2idx = dict((a, i) for i, a in enumerate(self.domain))
        self.idx2domain = dict((v, k) for k, v in self.domain2idx.items())
        
        self.intent2idx = dict((a, i) for i, a in enumerate(self.intent))
        self.idx2intent = dict((v, k) for k, v in self.intent2idx.items())
        
        self.slot2idx = dict((a, i) for i, a in enumerate(self.slot))
        self.idx2slot = dict((v, k) for k, v in self.slot2idx.items())
        
        self.inform2idx = dict((a, i) for i, a in enumerate(self.inform_da))
        self.idx2inform = dict((v, k) for k, v in self.inform2idx.items())
        
        self.request2idx = dict((a, i) for i, a in enumerate(self.request_da))
        self.idx2request = dict((v, k) for k, v in self.request2idx.items())
        
        self.da2idx = dict((a, i) for i, a in enumerate(self.da))
        self.idx2da = dict((v, k) for k, v in self.da2idx.items())
        
        self.dau2idx = dict((a, i) for i, a in enumerate(self.da_usr))
        self.idx2dau = dict((v, k) for k, v in self.dau2idx.items())
        
    def init_dim(self):
        self.s_dim = len(self.da)*2 + len(self.da_usr)*2 +len(self.inform_da) + len(self.request_da) + len(self.belief_domains) + 6*len(self.db_domains) + 1
        self.a_dim = len(self.da)
        self.a_dim_usr = len(self.da_usr)


class MultiWozConfig(Config):
    
    def __init__(self):
        self.domain = ['train', 'hotel', 'restaurant', 'attraction', 'taxi', 'general', 'booking']
        self.intent = ['inform', 'request', 'reqmore', 'bye', 'book', 'welcome', 'recommend', 'offerbook', 'nooffer', 'offerbooked', 'greet', 'select', 'nobook']
        self.slot = ['none', 'name', 'area', 'choice', 'type', 'price', 'ref', 'leave', 'addr', 'food', 'phone', 'day', 'arrive', 'depart', 'dest', 'id', 'post', 'people', 'stars', 'ticket', 'time', 'fee', 'car', 'internet', 'parking', 'stay']
        self.da = ['general-reqmore-none-none', 'general-bye-none-none', 'booking-inform-none-none', 'booking-book-ref-1', 'general-welcome-none-none', 'restaurant-inform-name-1', 'hotel-inform-choice-1', 'train-inform-leave-1', 'hotel-inform-name-1', 'train-inform-id-1', 'restaurant-inform-choice-1', 'train-inform-arrive-1', 'restaurant-inform-food-1', 'train-offerbook-none-none', 'restaurant-inform-area-1', 'hotel-inform-type-1', 'attraction-inform-name-1', 'restaurant-inform-price-1', 'attraction-inform-area-1', 'train-offerbooked-ref-1', 'hotel-inform-area-1', 'hotel-inform-price-1', 'general-greet-none-none', 'attraction-inform-choice-1', 'train-inform-choice-1', 'hotel-request-area-?', 'attraction-inform-addr-1', 'train-request-leave-?', 'taxi-inform-car-1', 'attraction-inform-type-1', 'taxi-inform-phone-1', 'restaurant-inform-addr-1', 'attraction-inform-fee-1', 'restaurant-request-food-?', 'attraction-inform-phone-1', 'hotel-inform-stars-1', 'booking-request-day-?', 'train-inform-dest-1', 'train-request-depart-?', 'train-request-day-?', 'attraction-inform-post-1', 'hotel-recommend-name-1', 'restaurant-recommend-name-1', 'hotel-inform-internet-1', 'train-request-dest-?', 'attraction-recommend-name-1', 'restaurant-inform-phone-1', 'train-inform-depart-1', 'hotel-inform-parking-1', 'train-offerbooked-ticket-1', 'booking-book-name-1', 'hotel-request-price-?', 'train-inform-ticket-1', 'booking-nobook-none-none', 'restaurant-request-area-?', 'booking-request-people-?', 'hotel-inform-addr-1', 'train-request-arrive-?', 'train-inform-day-1', 'train-inform-time-1', 'booking-request-time-?', 'restaurant-inform-post-1', 'booking-book-day-1', 'booking-request-stay-?', 'restaurant-request-price-?', 'attraction-request-type-?', 'attraction-request-area-?', 'booking-book-people-1', 'restaurant-nooffer-none-none', 'taxi-request-leave-?', 'hotel-inform-phone-1', 'taxi-request-depart-?', 'restaurant-nooffer-food-1', 'hotel-inform-post-1', 'booking-book-time-1', 'train-request-people-?', 'attraction-inform-addr-2', 'taxi-request-dest-?', 'restaurant-inform-name-2', 'hotel-select-none-none', 'restaurant-select-none-none', 'booking-book-stay-1', 'train-offerbooked-id-1', 'hotel-inform-name-2', 'hotel-nooffer-type-1', 'train-offerbooked-people-1', 'taxi-request-arrive-?', 'attraction-recommend-addr-1', 'attraction-recommend-fee-1', 'hotel-recommend-area-1', 'hotel-request-stars-?', 'restaurant-nooffer-area-1', 'restaurant-recommend-food-1', 'restaurant-recommend-area-1', 'attraction-recommend-area-1', 'train-inform-leave-2', 'hotel-inform-choice-2', 'attraction-nooffer-area-1', 'attraction-nooffer-type-1', 'hotel-nooffer-none-none', 'hotel-recommend-price-1', 'attraction-inform-name-2', 'hotel-recommend-stars-1', 'restaurant-inform-food-2', 'restaurant-recommend-price-1', 'train-select-none-none', 'attraction-inform-type-2', 'booking-inform-name-1', 'hotel-inform-type-2', 'hotel-request-type-?', 'hotel-request-parking-?', 'train-offerbooked-leave-1', 'attraction-select-none-none', 'hotel-select-type-1', 'taxi-inform-depart-1', 'hotel-inform-price-2', 'restaurant-recommend-addr-1', 'hotel-nooffer-area-1', 'hotel-inform-area-2', 'attraction-recommend-type-1', 'attraction-inform-type-3', 'hotel-nooffer-stars-1', 'hotel-nooffer-price-1', 'taxi-inform-dest-1', 'hotel-request-internet-?', 'taxi-inform-leave-1', 'hotel-recommend-type-1', 'restaurant-inform-choice-2', 'hotel-recommend-internet-1', 'restaurant-select-food-1', 'restaurant-nooffer-price-1', 'train-offerbook-id-1', 'hotel-recommend-parking-1', 'restaurant-inform-name-3', 'attraction-inform-addr-3', 'attraction-recommend-post-1', 'attraction-inform-choice-2', 'restaurant-inform-area-2', 'train-offerbook-leave-1', 'hotel-inform-addr-2', 'restaurant-inform-price-2', 'attraction-recommend-phone-1', 'hotel-select-type-2', 'train-offerbooked-arrive-1', 'attraction-inform-area-2', 'hotel-recommend-addr-1', 'restaurant-select-food-2', 'train-offerbooked-depart-1', 'attraction-select-type-1', 'train-offerbook-arrive-1', 'taxi-inform-arrive-1', 'restaurant-inform-post-2', 'attraction-inform-fee-2', 'restaurant-inform-food-3', 'train-offerbooked-dest-1', 'attraction-inform-name-3', 'hotel-select-price-1', 'attraction-request-name-?', 'train-inform-arrive-2', 'attraction-nooffer-none-none', 'train-inform-ref-1', 'booking-book-none-none', 'hotel-inform-stars-2', 'restaurant-select-price-1', 'hotel-inform-choice-3', 'attraction-inform-type-4']
        self.da_usr = ['general-welcome-none-none', 'restaurant-inform-food-1', 'train-inform-dest-1', 'train-inform-day-1', 'train-inform-depart-1', 'restaurant-inform-price-1', 'restaurant-inform-area-1', 'hotel-inform-stay-1', 'restaurant-inform-time-1', 'hotel-inform-type-1', 'general-bye-none-none', 'restaurant-inform-day-1', 'hotel-inform-day-1', 'restaurant-inform-people-1', 'attraction-inform-type-1', 'hotel-inform-price-1', 'hotel-inform-people-1', 'hotel-inform-stars-1', 'hotel-inform-area-1', 'train-inform-arrive-1', 'train-inform-people-1', 'attraction-inform-area-1', 'hotel-inform-name-1', 'train-inform-leave-1', 'hotel-inform-parking-1', 'hotel-inform-internet-1', 'restaurant-inform-name-1', 'attraction-inform-name-1', 'attraction-request-post-?', 'attraction-request-phone-?', 'attraction-request-addr-?', 'restaurant-request-addr-?', 'restaurant-request-phone-?', 'attraction-request-fee-?', 'train-request-ticket-?', 'taxi-inform-leave-1', 'taxi-inform-none-none', 'taxi-inform-depart-1', 'restaurant-inform-none-none', 'restaurant-request-post-?', 'taxi-inform-dest-1', 'train-request-time-?', 'hotel-inform-none-none', 'taxi-inform-arrive-1', 'train-request-ref-?', 'train-inform-none-none', 'hotel-request-addr-?', 'restaurant-request-ref-?', 'hotel-request-post-?', 'hotel-request-phone-?', 'train-request-id-?', 'hotel-request-ref-?', 'attraction-request-area-?', 'taxi-request-car-?', 'train-request-arrive-?', 'train-request-leave-?', 'attraction-inform-none-none', 'attraction-request-type-?', 'hotel-request-price-?', 'hotel-request-internet-?', 'hotel-request-parking-?', 'hotel-request-area-?', 'restaurant-request-price-?', 'restaurant-request-area-?', 'hotel-request-type-?', 'restaurant-request-food-?']
        self.data_file = 'annotated_user_da_with_span_full.json'
        self.ontology_file = 'value_set.json'
        self.db_domains = ['train', 'hotel', 'restaurant', 'attraction']
        self.belief_domains = ['train', 'hotel', 'restaurant', 'attraction', 'taxi']
        self.val_file = 'valListFile.json'
        self.test_file = 'testListFile.json'

        self.init_inform_request() # call this first!
        self.init_dict()
        self.init_dim()
        
        self.h_dim = 200
        self.hv_dim = 50 # for value function
        self.hu_dim = 200 # for user module
        self.eu_dim = 150
        self.max_ulen = 20
        self.alpha = 0.01
        self.hz_dim = 50 # for auto module
        self.hi_dim = 50 # for airl module
        
        self.mapping = {'restaurant': {'addr': 'address', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'price': 'pricerange'},
                        'hotel': {'addr': 'address', 'area': 'area', 'internet': 'internet', 'parking': 'parking', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'price': 'pricerange', 'stars': 'stars', 'type': 'type'},
                        'attraction': {'addr': 'address', 'area': 'area', 'fee': 'entrance fee', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'type': 'type'},
                        'train': {'id': 'trainID', 'arrive': 'arriveBy', 'day': 'day', 'depart': 'departure', 'dest': 'destination', 'time': 'duration', 'leave': 'leaveAt', 'ticket': 'price'},
                        'taxi': {'car': 'taxi_colors', 'phone': 'taxi_phone'}}
        self.map_inverse = {'restaurant': {'address': 'addr', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone', 'postcode': 'post', 'pricerange': 'price'},
                            'hotel': {'address': 'addr', 'area': 'area', 'internet': 'internet', 'name': 'name', 'parking': 'parking', 'phone': 'phone', 'postcode': 'post', 'pricerange': 'price', 'stars': 'stars', 'type': 'type'},
                            'attraction': {'address': 'addr', 'area': 'area', 'entrance fee': 'fee', 'name': 'name', 'phone': 'phone', 'postcode': 'post', 'type': 'type'},
                            'train': {'arriveBy': 'arrive', 'day': 'day', 'departure': 'depart', 'duration': 'time', 'destination': 'dest', 'leaveAt': 'leave', 'price': 'ticket', 'trainID': 'id'},
                            'taxi': {'arriveBy': 'arrive', 'car type': 'car', 'departure': 'depart', 'destination': 'dest', 'leaveAt': 'leave', 'phone': 'phone'}}   
