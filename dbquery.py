# -*- coding: utf-8 -*-
"""
@author: truthless 
"""

import random
import json
import numpy as np

def distance(str1, str2):
    m,n = len(str1)+1,len(str2)+1
    
    matrix = [[0]*n for i in range(m)]
    matrix[0][0] = 0
    for i in range(1,m):
        matrix[i][0] = matrix[i-1][0]+1
    for j in range(1,n):
        matrix[0][j] = matrix[0][j-1]+1
    
    for i in range(1,m):
        for j in range(1,n):
            if str1[i-1] == str2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j-1],matrix[i-1][j],matrix[i][j-1])+1
    
    return 1 - matrix[m-1][n-1] / max(m-1,n-1)

class DBQuery():
    def __init__(self, data_dir):
        # loading databases
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'taxi', 'police']
        self.dbs = {}
        for domain in domains:
            with open(data_dir + '/{}_db.json'.format(domain)) as f:
                self.dbs[domain] = json.load(f)
    
    def query(self, domain, constraints, noisy=False):
        """Returns the list of entities for a given domain
        based on the annotation of the belief state"""
        # query the db
        if domain == 'taxi':
            return [{'taxi_colors': random.choice(self.dbs[domain]['taxi_colors']), 
            'taxi_types': random.choice(self.dbs[domain]['taxi_types']), 
            'taxi_phone': ''.join(map(str,[random.randint(1, 9) for _ in range(10)]))}]
        if domain == 'police':
            return self.dbs['police']
        if domain == 'hospital':
            return self.dbs['hospital']
    
        found = []
        for i, record in enumerate(self.dbs[domain]):
            for key, val in constraints:
                if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
                    pass
                else:
                    if key not in record:
                        continue
                    if key == 'leaveAt':
                        try:
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                            if val1 > val2:
                                break
                        except (ValueError, IndexError):
                            continue
                    elif key == 'arriveBy':
                        try:
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                            if val1 < val2:
                                break
                        except (ValueError, IndexError):
                            continue
                    elif noisy and key in ['address', 'destination', 'departure', 'name']:
                        if distance(val.lower(), record[key].lower()) < 0.7:
                            break
                    else:
                        if val.lower() != record[key].lower():
                            break
            else:
                record['ref'] = f'{i:08d}'
                found.append(record)
    
        return found
    
    def pointer(self, turn, mapping, db_domains, noisy):
        """Create database pointer for all related domains."""        
        pointer_vector = np.zeros(6 * len(db_domains))
        for domain in db_domains:
            constraint = []
            for k, v in turn[domain].items():
                if k in mapping[domain]:
                    constraint.append((mapping[domain][k], v))
            entities = self.query(domain, constraint, noisy)
            pointer_vector = self.one_hot_vector(len(entities), domain, pointer_vector, db_domains)
    
        return pointer_vector

    def one_hot_vector(self, num, domain, vector, db_domains):
        """Return number of available entities for particular domain."""
        if domain != 'train':
            idx = db_domains.index(domain)
            if num == 0:
                vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num == 1:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num == 2:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num == 3:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num == 4:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num >= 5:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
        else:
            idx = db_domains.index(domain)
            if num == 0:
                vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num <= 2:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num <= 5:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num <= 10:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num <= 40:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num > 40:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    
        return vector  
