#!/bin/python
import sys
import random
import math
import time
from itertools import islice


class URM:
    def __init__(self, instructions, parse=True, max_iter=1000, registers=dict()):
        self._registers = dict()
        for key in registers.keys():
            self._set_register(key, registers[key])
        self._instructions = instructions
        if parse:
            self._instructions = URM.parse(instructions)
        self._ip = 0
        self._jumps = max_iter
        self._max_iter = max_iter

    def _get_register(self, n):
        if not (n in self._registers.keys()):
            self._set_register(n, 0)
        return (self._registers[n] & 0xFFFF) - 0x8000

    def _set_register(self, n, x):
        x = (x + 0x8000) & 0xFFFF
        self._registers[n] = x

    def _print_registers(self):
        print(self._registers)

    # instructions
    # Z(n) zeroes the contents of Rn
    def ZERO(self, n):
        self._set_register(n, 0)

    # S(n) adds 1 to the contents of Rn
    def INC(self, n):
        self._set_register(n, self._get_register(n) + 1)

    # D(n) adds 1 to the contents of Rn
    def DEC(self, n):
        self._set_register(n, self._get_register(n) - 1)

    # C(n,m) copies the contents of Rn to Rm overwriting the contents of Rm
    def COPY(self, n, m):
        self._set_register(m, self._get_register(n))

    # A(n,m) ADDs n to m and set result in n
    def ADD(self, n, m):
        self._set_register(n, self._get_register(n) + self._get_register(m))

    # M(n,m) SUBTRACTs m from n and set result in n
    def SUB(self, n, m):
        self._set_register(n, self._get_register(n) - self._get_register(m))

    # T(n,m) Multipliess m from n and set result in n
    def MUL(self, n, m):
        self._set_register(n, self._get_register(n) * self._get_register(m))

    # DIV(n,m) divides n by m and set result in n
    def DIV(self, n, m):
        if self._get_register(m) == 0:
            self._set_register(n, 0)
        else:
            self._set_register(n, self._get_register(n) // self._get_register(m))

    # J(n,m,k) which means ... if the contents of Rn & Rm are the same then Jump to instruction number k, otherwise, continue on to the next instruction of the program.
    def JEREL(self, n,m,k):
        if self._get_register(n) == self._get_register(m):
            self._ip = self._ip + k
            self._jumps -= 1
            if self._jumps == 0:
                #print("too many jumps, forcing stop by jumping to invalid pos")
                self._ip = 99999
                self._set_register(1, -0x8000)

    def JNEREL(self, n,m,k):
        if self._get_register(n) != self._get_register(m):
            self._ip = self._ip + k
            self._jumps -= 1
            if self._jumps == 0:
                #print("too many jumps, forcing stop by jumping to invalid pos")
                self._ip = 99999
                self._set_register(1, -999999)

    # J(n,m,k) which means ... if the contents of Rn & Rm are the same then Jump to instruction number k, otherwise, continue on to the next instruction of the program.
    def JEABS(self, n,m,k):
        if self._get_register(n) == self._get_register(m):
            self._ip = k
            self._jumps -= 1
            if self._jumps == 0:
                #print("too many jumps, forcing stop by jumping to invalid pos")
                self._ip = 99999
                self._set_register(1, -0x8000)

    def JNEABS(self, n,m,k):
        if self._get_register(n) != self._get_register(m):
            self._ip = k
            self._jumps -= 1
            if self._jumps == 0:
                #print("too many jumps, forcing stop by jumping to invalid pos")
                self._ip = 99999
                self._set_register(1, -999999)

    def execute(self):
        while self._ip < len(self._instructions):
            if  self._ip < 0:
                self._ip = 0
            cmd = self._instructions[self._ip]
            try:
                self._ip=self._ip+1
                if cmd[0].upper() in ['Z', 'ZERO', '0']:
                    self.ZERO(cmd[1])
                elif cmd[0].upper() in ['S', 'INC', '++']:
                    self.INC(cmd[1])
                elif cmd[0].upper() in ['D', 'DEC', '--']:
                    self.DEC(cmd[1])
                elif cmd[0].upper() in ['C', 'COPY', 'MOVE', 'MOV', ':=']:
                    self.COPY(cmd[1], cmd[2])
                elif cmd[0].upper() in ['+', 'ADD', 'SUM']:
                    self.ADD(cmd[1], cmd[2])
                elif cmd[0].upper() in ['-', 'SUB']:
                    self.SUB(cmd[1], cmd[2])
                elif cmd[0].upper() in ['*', 'MUL', 'MULT']:
                    self.MUL(cmd[1], cmd[2])
                elif cmd[0].upper() in ['/', 'DIV']:
                    self.DIV(cmd[1], cmd[2])
                elif cmd[0].upper() in ['J', 'JE', 'JUMPEQUAL']:
                    self.JEREL(cmd[1], cmd[2], cmd[3])
                elif cmd[0].upper() in ['K', 'JNE', 'JUMPNOTEQUAL']:
                    self.JNEREL(cmd[1], cmd[2], cmd[3])
                elif cmd[0].upper() in ['JEADDR', 'JEADR', 'JEA']:
                    self.JEABS(cmd[1], cmd[2], cmd[3])
                elif cmd[0].upper() in ['JNEADDR', 'JNEADR', 'JNEA', 'JNA']:
                    self.JNEABS(cmd[1], cmd[2], cmd[3])
            except Exception as err:
                print(self._instructions)
                print(cmd)
                self._print_registers()
                print(err)
                exit(-1)

        return {"result":self._get_register(1),
                "registers":self._registers,
                "size_registers":max(self._registers.keys()),
                "size": len(self._instructions),
                "jumps": (self._max_iter - self._jumps)}

    @staticmethod
    def parse(code):
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
            elem = elem.split("(")
            symbol = elem[0]
            elem = elem[1]
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
    
    @staticmethod
    def unparse(l):
        ret = ""
        for i in l:
            ret += i[0] + "(" + ",".join([str(j) for j in i[1:]]) +") "
        return ret

class Individual:

    def __init__(self, max_genes, genes=None):
        self._max_genes = max_genes
        self._value = None
        if genes:
            self._genes = genes
        else:
            self._genes = self.make_random_genes()
    
    def __len__(self):
        return len(self._genes)

    def instatiate_child(self, genes=None):
        pass

    def _gene_getter(self):
        pass

    def _evaluator(self):
        pass

    def print(self):
        pass

    def make_random_genes(self):
        # make genes immutable always
        return tuple([self._gene_getter() for j in range(random.randrange(3,self._max_genes+1))])

    def mate(self, partner):
        i1 = self._genes
        i2 = partner._genes
        offspring =  [i1[i] for i in range(random.randrange(0, len(i1)))] + [i2[i] for i in range(random.randrange(0, len(i2)), len(i2))]
        return self.instatiate_child(tuple(offspring))

    def mate_old(self, partner):
        i1 = self._genes
        i2 = partner._genes
        if len(i1) > len(i2):
            inds = [i2, i1]
        else:
            inds = [i1,i2]
        ret = []
        for i in range(len(inds[0])):
            ret.append(random.choice(inds)[i])
        if len(inds[1]) > len(inds[0]) and random.choice([True, False]):
            ret += inds[1][len(inds[0]):]
        return ret


    def mutate(self):
        ret = self.instatiate_child(tuple(self._mutate_genes()))
        return ret

    def _mutate_genes(self):
        if random.uniform(0, 1) < 0.85:
            return self.copy_genes()
        oper = random.randrange(0, 6)
        if oper == 0:
            return self.shorten()
        if oper == 1: 
            return self.widen()
        if oper == 2: 
            return self.change_gene()
        if oper == 3: 
            return self.swap_gene()
        if oper == 4: 
            return self.reverse()
        else:
            return self.make_random_genes()
    
    def copy_genes(self):
        #return [i for i in self._genes]
        return self._genes

    def reverse(self):
        ret = [i for i in reversed(self._genes)]
        return ret

    def shorten(self):
        if len(self._genes) <= 3:
            return self.change_gene()
        pos = random.randrange(len(self._genes))
        ret = [self._genes[i] for i in range(pos)] + [self._genes[i] for i in range(pos+1, len(self._genes))]
        if (len(self._genes) <= len(ret)):
            print ("ERROR shorten", len(self._genes), len(ret))
        return ret

    def widen(self):
        if len(self._genes) >= self._max_genes:
            return self.change_gene()
        pos = random.randrange(len(self._genes))
        ret = [self._genes[i] for i in range(pos)] + [self._gene_getter()] + [self._genes[i] for i in range(pos, len(self._genes))]
        if (len(self._genes) >= len(ret)):
            print ("ERROR widen", len(self._genes), len(ret))
        return ret

    def change_gene(self):
        pos = random.randrange(len(self._genes))
        ret = [self._genes[i] for i in range(pos)] + [self._gene_getter()] + [self._genes[i] for i in range(pos+1, len(self._genes))]
        if (len(self._genes) != len(ret)):
            print ("ERROR change_gene", len(self._genes), len(ret))
        return ret

    def swap_gene(self):
        pos1 = random.randrange(len(self._genes))
        pos2 = random.randrange(len(self._genes))
        ret = [i for i in self._genes]
        ret[pos1],ret[pos2] = ret[pos2],ret[pos1]
        if (len(self._genes) != len(ret)):
            print ("ERROR change_gene", len(self._genes), len(ret))
        return ret

    def evaluate(self):
        if not self._value:
            self._value = self._evaluator()
        return self._value
    
    def __repr__(self):
        return "Individual " + str(self._genes)

    def __eq__(self, other):
        if isinstance(other, Individual):
            return self.__repr__() == other.__repr__()
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())


class GA:
    def __init__(self, base_individual : Individual, pop_size, is_tournament=True):
        self._pop_size = pop_size
        self.select = self.roulete_select
        if is_tournament:
            self.select = self.tournament_select
            #self.select = self.single_tournament_select
        self._base_individual = base_individual
        self._pop = self._build_pop()
        self._total_score = None
        self._sorted = False
        self._best = False
        self._all_best = False


    def _build_pop(self):
        return [self._base_individual.instatiate_child() for i in range(self._pop_size)]

    def tournament_select(self, k=15, p=0.75):
        if k < 1:
            k = int(self._pop_size / k)
        if not self._sorted:
            self._sorted = [sorted(self._pop[n:n+k], key=lambda x: x.evaluate(), reverse=True) for n in range((self._pop_size // k) +1) if len(self._pop[n:n+k])>0]
        chosen = random.choice(self._sorted)
        x = random.uniform(0,1)
        n = min(len(chosen)-1, int(math.log(1 - x)/math.log(1 - p)))
        return chosen[n]

    def single_tournament_select(self, k=1, p=0.5):
        return self.tournament_select(self._pop_size, p)

    def roulete_select(self):
        if not self._total_score:
            self._total_score = sum([ind.evaluate() for ind in self._pop])
        value = random.uniform(0, self._total_score)
        ind = self._pop[0]
        ind_sum = 0
        for ind in self._pop:
            ind_sum += ind.evaluate()
            if ind_sum > value:
                return ind
        return self._pop[len(self._pop)-1]

    def nextgen(self):
        keep_best = int(0.01 * self._pop_size)
        best = self.get_best_N(keep_best, unique=True)
        rest = [self.select().mate(self.select()).mutate() for i in range(len(best), self._pop_size)]
        self._pop = best + rest
        random.shuffle(self._pop)
        if len(self._pop) != self._pop_size:
            print("ERROR: pop size is wrong", len(self._pop), self._pop_size)
        self._sorted = False
        self._total_score = None
        self._best = False
        self._all_best = False
        
    def get_best(self):
        if not self._best:
            self._best = max(self._pop, key=lambda x: x.evaluate())
        return self._best

    def get_best_count(self):
        best = self.get_best()
        return len([i for i in self._pop if i.evaluate() == best.evaluate()])

    def get_best_N(self, n, unique=False):
        if not self._all_best:
            self._all_best = sorted(self._pop, key=lambda x : x.evaluate(), reverse=True)
        best = self._all_best
        if unique:
            best = sorted(set(self._all_best), key=lambda x : x.evaluate(), reverse=True)
        return best[:n]

def get_random_oper(regs, max_jump, opers=['ZERO', 'INC', 'COPY', 'JE', 'DEC', 'JNE', 'ADD', 'SUB', 'MUL', 'JEA', 'JNA']):
    oper = random.choice(opers)
    reg1 = random.randrange(1, regs+1)
    if oper in ['Z', 'S', 'D', 'ZERO', 'INC', 'DEC']:
        return [oper, reg1]
    elif oper in ['C', '-', '+', '*', 'COPY', 'ADD', 'SUB', 'MUL', '/', 'DIV']:
        reg2 = random.randrange(1, regs+1)
        return [oper, reg1, reg2]
    elif oper in ['J', 'K', 'JE', 'JNE']:
        reg2 = random.randrange(1, regs+1)
        jump = random.randrange(-max_jump, max_jump+1)
        return [oper, reg1, reg2, jump]
    elif oper in ['JEA', 'JNA']:
        reg2 = random.randrange(1, regs+1)
        jump = random.randrange(1, max_jump+1)
        return [oper, reg1, reg2, jump]
    else:
        raise Exception("Invalid operator, should not happen")

def eval_oper(code, rangeA, rangeB, oper, max_iter, isprint=False):
    total_error = 0
    total_error_sqr = 0
    total_matches = 0
    total_jumps = 0
    total_registers = 0
    total_relative_error = 0.0
    count = 0
    for a in rangeA:
        for b in rangeB:
            m = oper(a,b)
            registers = {1:a, 2:b}
            urm = URM(code, False, max_iter, registers)
            result_struct = urm.execute()
            result = result_struct["result"]
            size_registers = result_struct["size_registers"]
            jumps = result_struct["jumps"]
            total_jumps += jumps
            total_registers += size_registers
            error_sqr = (m - result)**2
            error = abs(m-result)
            total_error_sqr += error_sqr
            total_error += error
            total_relative_error += (1.0*error)/max(m, 1)
            count += 1
            if result == m:
                total_matches += 1
    avg_error = (1.0*total_error)/count
    avg_relative_error = (1.0*total_relative_error)/count
    avg_jumps = (1.0*total_jumps)/count
    avg_registers = (1.0*total_registers)/count
    stddev = math.sqrt(total_error_sqr)/count
    variance = total_error_sqr/(count-1)
    factor = len(code) * 0.001 + total_error * 0 + stddev * 0 + variance * 0 + avg_error * 0 + avg_relative_error * 1.0
    # minimize jumps and registers
    factor += avg_jumps * 0.0001 + avg_registers * 0.001
    score = 1.0/factor
    # maximize matches
    #score += total_matches * 0
    # cap to 100K
    score = min(100000, score)
    if isprint:
        print("matched %d out of %d times" % (total_matches, count))
        print(" total error %d, stddev %f, variance %f" % (total_error, stddev, variance))
        print(" Average error %f, Average Relative Error %f" % (avg_error, avg_relative_error))
        print(" size %d, registers %f, jumps %f" % (len(code), avg_registers, avg_jumps))
        print(" final score: %f" % (score))
    return score

class URMIndividual(Individual):
    def __init__(self, max_genes, rangea, rangeb, oper, genes=None):
        super().__init__(max_genes, genes=genes)
        self._rangea = rangea
        self._rangeb = rangeb
        self._oper = oper

    def instatiate_child(self, genes=None):
        pass

    def _gene_getter(self):
        return get_random_oper(5, 30)

    def _evaluator(self):
        return eval_oper(self._genes, self._rangea, self._rangeb, self._oper, 30)

    def print(self):
        return eval_oper(self._genes, self._rangea, self._rangeb, self._oper, 30, True)

    def __repr__(self):
        return URM.unparse(self._genes)



class AddIndividual(URMIndividual):
    def __init__(self, genes=None):
        super().__init__(30, range(0,10), range(0,10), lambda x, y : x+y, genes=genes)

    def instatiate_child(self, genes=None):
        return AddIndividual(genes)

    def _gene_getter(self):
        return get_random_oper(5, 30, ['ZERO', 'INC', 'COPY', 'JE', 'DEC', 'JNE'])

class SubIndividual(URMIndividual):
    def __init__(self, genes=None):
        super().__init__(30, range(0,10), range(0,10), lambda x, y : x-y, genes=genes)

    def instatiate_child(self, genes=None):
        return SubIndividual(genes)
        
    def _gene_getter(self):
        return get_random_oper(5, 30, ['ZERO', 'INC', 'COPY', 'JE', 'DEC', 'JNE'])

class MulIndividual(URMIndividual):
    def __init__(self, genes=None):
        super().__init__(30, range(0,6), range(0,6), lambda x, y : x*y, genes=genes)

    def instatiate_child(self, genes=None):
        return MulIndividual(genes)
        
    def _gene_getter(self):
        return get_random_oper(5, 30, ['ZERO', 'INC', 'COPY', 'JE', 'DEC', 'JNE', 'JEA', 'JNA'])
    
    def _evaluator(self):
        return eval_oper(self._genes, self._rangea, self._rangeb, self._oper, 300)

    def print(self):
        return eval_oper(self._genes, self._rangea, self._rangeb, self._oper, 300, True)


class NineIndividual(URMIndividual):
    def __init__(self, genes=None):
        super().__init__(30, range(1), range(1), lambda x, y : 9, genes=genes)

    def instatiate_child(self, genes=None):
        return NineIndividual(genes)

    def _gene_getter(self):
        return get_random_oper(5, 30, ['ZERO', 'INC', 'COPY', 'JEA'])

    def _evaluator(self):
        urm = URM(self._genes, False, 20)
        result_struct = urm.execute()
        result = result_struct["result"]
        size = len(self._genes) * 0.001
        score = 1/((9-result)**2 + size)
        score = min(100000, score)
        return score

    def print(self):
        print(self)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            registers = [int(x) for x in sys.argv[2].split()]
            registers = dict(zip(range(1, len(registers)+1), registers))
            print(registers)
            urm = URM(sys.argv[1], True, 10000, registers)
        else:
            urm = URM(sys.argv[1])
        result = urm.execute()
        print(result)
        exit(0)
    print("No command line args provided. will try to run a genetic algorithm to guess the best solution")

    # build valid cmds
    POP_SIZE=100

    ga = GA(MulIndividual(), POP_SIZE)
    ts0 = time.time()
    ts = ts0
    i = 0
    while True:
        i = i+1
        ga.nextgen()
        ts2 = time.time()
        if (ts2 - ts >= 30):
            ts = ts2
            print ('Results after', i, 'generations')
            result = ga.get_best()
            result.print()
            all_best = [i for i in ga.get_best_N(10, True) if i.evaluate() == result.evaluate()]
            print('Found %d solutions with size %d and result %f' % (ga.get_best_count(), len(result), result.evaluate()))
            for ind in all_best:
                print(" %f:" % ind.evaluate())
                print("    %s" % repr(ind))
            print('===================================')
