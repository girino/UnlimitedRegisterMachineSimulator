#!/bin/python
import sys

class URM:
    def __init__(self, instructions, parse=True):
        self._registers = dict()
        self._instructions = instructions
        if parse:
            self._instructions = self._parse(instructions)
        self._ip = 0

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
            self._ip = k

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
            print(cmd)
            self._print_registers()
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

if __name__ == "__main__":
    urm = URM(sys.argv[1])
    print(urm.execute())
        
