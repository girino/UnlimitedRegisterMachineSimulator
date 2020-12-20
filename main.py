#!/bin/python
import sys
import random
import math
from tqdm import tqdm


MAX_ITER=10

class URM:
    def __init__(self, instructions, parse=True):
        self._registers = dict()
        self._instructions = instructions
        if parse:
            self._instructions = self._parse(instructions)
        self._ip = 0
        self._jumps = 0

    def _get_register(self, n):
        if not (n in self._registers.keys()):
            self._registers[n] = 0
        return self._registers[n]

    def _zero_registers(self):
        self._registers = dict()

    def _set_register(self, n, x):
        self._registers[n] = x

    def _print_registers(self):
        print(self._registers)

    # instructions
    # Z(n) zeroes the contents of Rn
    def Z(self, n):
        self._set_register(n, 0)

    # S(n) adds 1 to the contents of Rn
    def S(self, n):
        self._set_register(n, self._get_register(n) + 1)

    # C(n,m) copies the contents of Rn to Rm overwriting the contents of Rm
    def C(self, n, m):
        self._set_register(m, self._get_register(n))

    # J(n,m,k) which means ... if the contents of Rn & Rm are the same then Jump to instruction number k, otherwise, continue on to the next instruction of the program.
    def J(self, n,m,k):
        if self._get_register(n) == self._get_register(m):
            self._ip = k-1
            self._jumps += 1
            if self._jumps == MAX_ITER:
                #print("too many jumps, forcing stop by jumping to invalid pos")
                self._ip = 99999
                self._set_register(1, -1)

    def execute(self):
        self._zero_registers()
        while self._ip < len(self._instructions):
            cmd = self._instructions[self._ip]
            self._ip=self._ip+1
            if cmd[0] == 'Z':
                self.Z(cmd[1])
            if cmd[0] == 'S':
                self.S(cmd[1])
            if cmd[0] == 'C':
                self.C(cmd[1], cmd[2])
            if cmd[0] == 'J':
                self.J(cmd[1], cmd[2], cmd[3])
            #print(cmd)
            #self._print_registers()
        return self._get_register(1)

    def _parse(self, code):
        ret = []
        # remove spaces
        code = code.replace(" ", "")
        # split on closing brackets
        ins = code.split(")")
        # for each instruction
        for elem in ins:
            if not elem:
                continue
            # get starting symbol
            symbol = elem[0]
            # remove opening brackets
            elem = elem[2:]
            # split on commas
            params = elem.split(",")
            int_params = []
            # make params integer
            for i in params:
                int_params = int_params + [int(i)]
            # build output code
            out = [symbol] + int_params
            ret = ret + [out]
        return ret
    
def unparse(l):
    ret = ""
    for i in l:
        ret += i[0] + "(" + ",".join([str(j) for j in i[1:]]) +") "
    return ret

class GA:
    def __init__(self, pop_size, genes, num_genes):
        self._pop_size = pop_size
        self._genes = genes
        self._num_genes = num_genes
        self._pop = self._build_pop()
        self.evaluate_all()

    def _build_pop(self):
        size = random.randrange(3,self._num_genes+1)
        return [[random.choice(self._genes) for j in range(size)] for i in range(self._pop_size)]
    
    def evaluate(self, individual, isprint=False):
        urm = URM(individual, False)
        result = urm.execute()
        size_weight = math.exp(len(individual)) * 0.0024
       # cap at 100000
        if isprint:
            print("result =", result, 'size weight', size_weight, 'final score = ', 1.0/(abs(9.0 - result) + size_weight))
        return min(100000.0, 1.0/(abs(9.0 - result) + size_weight))

    def evaluate_all(self):
        self._scores = [self.evaluate(ind) for ind in self._pop]
        self.score_sum = sum(self._scores)

    def roulete_select(self):
        value = random.uniform(0, self.score_sum)
        pos = 1
        sum = self._scores[0]
        ind = self._pop[0]
        while sum < value and pos < len(self._scores):
            sum += self._scores[pos]
            ind = self._pop[pos]
            pos += 1
        return ind

    def nextgen(self):
        best = self.get_all_best()
        rest = [self.mutate(self.mate(self.roulete_select(), self.roulete_select())) for i in range(self._pop_size-len(best))]
        self._pop = best + rest
        self.evaluate_all()

    def get_best(self):
        maxs = 0.0
        maxd = self._pop[0]
        for i in range(self._pop_size):
            if self._scores[i] > maxs:
                maxd = self._pop[i]
                maxs = self._scores[i]
        return (maxd, maxs)

    def get_all_best(self):
        maxd, maxs = self.get_best()
        ret = []
        for i in range(self._pop_size):
            if self._scores[i] == maxs:
                if not any([self.compare_individual(self._pop[i], ind) for ind in ret]):
                    ret.append(self._pop[i])
        return ret

    def compare_oper(self, o1, o2):
        return len(o1) == len(o2) and all([o1[i] == o2[i] for i in range(len(o1))])

    def compare_individual(self, i1, i2):
        return len(i1) == len(i2) and all([self.compare_oper(i1[i], i2[i]) for i in range(len(i1))])
        

    def mutate(self, individual):
        if random.uniform(0, 1) < 0.80:
            return individual
        oper = random.randrange(0, 4)
        if oper == 0:
            return self.shorten(individual)
        if oper == 1: 
            return self.widen(individual)
        if oper == 2: 
            return self.change_gene(individual)
        else:
            size = random.randrange(3,self._num_genes+1)
            return [random.choice(self._genes) for j in range(size)]

    def shorten(self, individual):
        if len(individual) <= 3:
            return self.change_gene(individual)
        pos = random.randrange(len(individual))
        return [individual[i] for i in range(pos)] + [individual[i] for i in range(pos+1, len(individual))]

    def widen(self, individual):
        if len(individual) >= 30:
            return self.change_gene(individual)
        pos = random.randrange(len(individual))
        return [individual[i] for i in range(pos)] + [random.choice(self._genes)] + [individual[i] for i in range(pos, len(individual))]

    def change_gene(self, individual):
        pos = random.randrange(len(individual))
        return [individual[i] for i in range(pos)] + [random.choice(self._genes)] + [individual[i] for i in range(pos+1, len(individual))]
    
    def mate(self, i1, i2):
        if len(i1) > len(i2):
            inds = [i2, i1]
        else:
            inds = [i1,i2]
        ret = []
        for i in range(len(inds[0])):
            ret.append(random.choice(inds)[i])
        if len(inds[1]) > len(inds[0]) and random.choice([True, False]):
            ret += inds[1][len(inds[0]):]
        #print (i1)
        #print (i2)
        #print ('->', ret)
        return ret



def build_all_c(regs):
    ret = []
    for i in range(regs):
        for j in range(regs):
            ret = ret + [['C', i+1, j+1]]
    return ret

def build_all_j(regs, inst_size):
    ret = []
    for i in range(regs):
        for j in range(regs):
            for k in range(inst_size+1):
                ret = ret + [['J', i+1, j+1, k+1]]
    return ret

def build_all_simple(regs, symbol):
    ret = []
    for i in range(regs):
        ret = ret + [[symbol, i+1]]
    return ret

if __name__ == "__main__":

    if len(sys.argv) > 1:
        urm = URM(sys.argv[1])
        result = urm.execute()
        print(result)
        exit(0)
    print("No command line args provided. will try to run a genetic algorithm to guess the best solution")

    # build valid cmds
    regs = 3
    inst_size = 9
    all_valid = build_all_simple(1, 'Z') + build_all_simple(1, 'S') + build_all_c(regs) + build_all_j(regs+1, inst_size)

    ga = GA(1500, all_valid, 15)
    for i in range(10000):
        ga.nextgen()
        if (i % 10) == 0:
            print ('Results after', i, 'iterations')
            best = ga.get_best()
            print("Best Scores")
            ga.evaluate(best[0], True)
            print('Best solutions')
            for best in ga.get_all_best():
                print(" ", unparse(best))
            print('===================================')
