# UnlimitedRegisterMachineSimulator

So, this all started with a problem proposed in facebook. Then it grew into a GA to solve the problem. Then it grew into a generic GP for the proposed theoretical machine. So here it is. A GA/GP to solve generic problems using a theoretical Unlimited Register Machine.

No docs provided, use at your own risk.

## The original problem

Here’s a little problem for Christmas and you won’t be able to cheat and look the answer up online, (I know because I tried), it took a friend and I several days to work out the answer!
An Unlimited Register Machine (URM) has an unlimited number of registers labelled R1, R2, R3, ..., Rn. The contents of these registers are called r1, r2, r3, ... etc. & all contain zero by default.
All registers can hold +ve natural numbers & zero i.e. 0,1,2,3, ... etc. A URM program is read & executed top down. It consists of a set of instructions numbered 1,2,3, ... n & ends when it reaches an instruction number for which there is no instruction e.g. go to 9, but there is no 9 i.e. the program only has 8 instructions.
There are only four instructions available to a URM program:-
Z(n) zeroes the contents of Rn
S(n) adds 1 to the contents of Rn
C(n,m) copies the contents of Rn to Rm overwriting the contents of Rm
J(n,m,k) which means ... if the contents of Rn & Rm are the same then Jump to instruction number k, otherwise, continue on to the next instruction of the program.
The output of a URM program is always contained in R1.
Here's an example.
1 S(1)
2 S(1)
3 S(1)
4 J(1,1,1)
This is an infinite loop that moves through the three times table.
Here's another example.
1 S(2)
2 S(2)
3 S(2)
4 C(2,1)
This ends because there are no more instructions & outputs 3.
1 S(5)
2 S(5)
3 J(4,5,10)
4 S(4)
5 S(4)
6 J(1,1,3)
This ends because of the Jump to the non-existent instruction number 10 & outputs zero.
The problem is to devise a URM program of fewer than 9 instructions, which outputs the number 9, all registers = 0 at the start. If we could use nine instructions, then this could easily be achieved with:-
1 S(1)
2 S(1)
...
8 S(1)
9 S(1)
Good luck, have fun.