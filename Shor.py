import math, fractions, contfrac
from qiskit import execute, ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.tools.monitor import job_monitor
from Shor_modules import cmult_mod_N
from gmpy2 import mpfr
from QuantumFourierTransform import qft, inverse_qft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as ax
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit import Aer

from teste import to_quantum


def getAngle(a, N):
    """convert the number a to a binary string with length N"""
    s=bin(int(a))[2:].zfill(N)
    angle = 0
    for i in range(0, N):
        """if the digit is 1, add the corresponding value to the angle"""
        if s[N-1-i] == '1':
            angle += math.pow(2, -(N-i))
    angle *= np.pi
    return angle

def get_factors(factors: set, x_value, t_upper, N, a):

    if x_value <= 0:
        return False

    """Compute continued fractions coefficients for x"""
    b = list(contfrac.continued_fraction((x_value, pow(2, t_upper))))

    """Compute and test fractions using the computed coefficient"""
    for i in range(0, len(b)):

        """ Calculate the CF using the first i coefficients """
        aux = 0
        for j in range(i, 0, -1):
            aux = 1 / (b[j] + aux)
        aux = aux + b[0]

        """Get the denominator from the value obtained"""
        frac = fractions.Fraction(aux).limit_denominator()
        den = frac.denominator

        if (den % 2) == 1:
            """if denominator is odd continue searching"""
            continue

        """Maximum representable integer in 32 bit system is 2^31"""
        if den < 62:
            exponential = pow(a, (den / 2))
        else:
            return False

        """ Check if the value is too big or not """
        if math.isinf(exponential) == 1:
            return False

        flag = False
        new_guess_plus = int(exponential + 1)
        new_guess_minus = int(exponential - 1)
        aux_1 = math.gcd(new_guess_plus, N)
        aux_2 = math.gcd(new_guess_minus, N)
        if aux_1 != 1 and aux_1 != N:
            factors.add(aux_1)
            flag = True

        if aux_2 != 1 and aux_2 != N:
            factors.add(aux_2)
            flag = True


        """ Check if the factors found are trivial factors or are the desired
        factors """

        if (aux_1 == 1 or aux_1 == N) and (aux_2 == 1 or aux_2 == N):
            if i == len(b):
                print('The continued fractions found exactly x_final/(2^(2n)) , leaving funtion\n')
                return False
        else:
            return flag

def shor4n2(N: int, a: int, backend, shots = 2048, print_statistics = False)->set:

    if print_statistics:
        stats = open('shor_stats.txt', 'w')

    """Number of bits necessary to represent N"""
    n = math.ceil(math.log(N, 2))
    #n = int(math.ceil(math.log(N, 2))/2)
    print('Total number of qubits used: {0}\n'.format(4 * n + 2))

    up_reg = QuantumRegister( 2*n, name='x')
    down_reg = QuantumRegister(2 * n + 2, name='w')
    up_classic = ClassicalRegister(2*n)

    """ Create Quantum Circuit """
    circuit = QuantumCircuit(up_reg, down_reg, up_classic, name='Shor')

    """ Initialize down register to |1> and create maximal superposition in top register """
    circuit.h(up_reg)
    circuit.x(down_reg[0])

    """ Apply the multiplication gates to create the exponentiation """
    aux_aux = QuantumRegister(n+2)
    aux_up_reg = QuantumRegister(2*n)
    aux_down_reg = QuantumRegister(n)
    aux_circuit = QuantumCircuit(aux_up_reg, aux_down_reg, aux_aux, name='U_x,N')
    for i in range(0, 2*n):
        cmult_mod_N(aux_circuit, aux_up_reg[i], aux_down_reg, aux_aux, mpfr(pow(a, pow(2, i))), N, n)
    q_args = np.array(up_reg).tolist()+np.array(down_reg).tolist()
    circuit.append(aux_circuit.to_instruction(), qargs=q_args)

    """ Apply inverse QFT """

    circuit.append(inverse_qft(2*n), up_reg)

    """ Measure the top qubits, to get x value"""
    circuit.measure(up_reg, up_classic)

    """ Simulate the created Quantum Circuit """
    simulation = execute(circuit, backend=backend, shots=shots, optimization_level=3, memory=True)
    job_monitor(simulation)
    sim_result = simulation.result()
    counts_result = sim_result.get_counts(circuit)
    circuit.draw(output='mpl')
    plt.show()


    if print_statistics:
        dict={}
        for i in range(256):
            dict[i] = 0
        for i in range(0, len(counts_result)):
            print('Result \"{0}\" ({3}) happened {1} times out of {2}'.format(list(sim_result.get_counts().keys())[i],
                                                                              list(sim_result.get_counts().values())[i],
                                                                              shots,
                                                                              int(list(sim_result.get_counts().keys())[
                                                                                      i],
                                                                                  2)), file=stats)
            dict[int(list(sim_result.get_counts().keys())[i], 2)] = list(sim_result.get_counts().values())[i]
        print(dict, file=stats)
        for i in range(0,256,4):
            for k in range(1,4):
                dict[i] += dict[i+k]
                dict.pop(i+k)
        plot_histogram(dict, figsize=(28, 6), bar_labels=False, title='Measurement for N=15, g=2 (2048 samples)')
        plt.xticks([0, 16, 32, 48, 63], [0, 64, 128, 192, 255])
        plt.show()

    """ Initialize this variable """
    prob_success = 0
    list_of_factors = set()
    """ For each simulation result, print proper info to user and try to calculate the factors of N"""

    for i in range(0, len(counts_result)):

        """ Get the x_value from the final state qubits """
        output_desired = list(sim_result.get_counts().keys())[i]
        x_value = int(output_desired, 2)
        prob_this_result = 100 * (int(list(sim_result.get_counts().values())[i])) / (shots)

        if get_factors(list_of_factors, int(x_value), int(2 * n), int(N), int(a)):
            prob_success = prob_success + prob_this_result

    if print_statistics:
        if len(list_of_factors):
            print("\nUsing a={0}, found the factors {3} for N={1} in {2:.4f} % of the cases\n".format(a,N,prob_success,
                                                                                         list_of_factors), file=stats)
        else:
            print("\nUnable to find factors of N={1} using a={0}\n".format(a, N), file=stats)
        stats.close()

    """In case only one factor was discovered calculate the other"""
    if len(list_of_factors) == 1:
        for a in list_of_factors:
            factor = a
        list_of_factors.add(int(N/factor))
    return list_of_factors




def shor2n3(N: int, a: int, backend, shots = 2048, print_statistics = False)->set:

    if print_statistics:
        stats = open('shor_stats.txt', 'w')

    """Number of bits necessary to represent N"""
    n = math.ceil(math.log(N, 2))
    # n = int(math.ceil(math.log(N, 2))/2)
    print('Total number of qubits used: {0}\n'.format(2 * n + 3))

    aux = QuantumRegister(n + 2, name='b')
    up_reg = QuantumRegister(1, name='x')
    down_reg = QuantumRegister(n, name='w')
    up_classic = ClassicalRegister(2 * n)
    c_aux = ClassicalRegister(1)

    """ Create Quantum Circuit """
    circuit = QuantumCircuit(up_reg, down_reg, aux, up_classic, c_aux, name='Shor')

    """ Initialize down register to |1> and create maximal superposition in top register """
    circuit.x(down_reg[0])

    """ Cycle to create the Sequential QFT, measuring qubits and applying the right gates according to measurements """
    for i in range(0, 2*n):
        """reset the top qubit to 0 if the previous measurement was 1"""
        circuit.x(up_reg).c_if(c_aux, 1)
        circuit.h(up_reg)
        cmult_mod_N(circuit, up_reg[0], down_reg, aux, mpfr(a**(2**(2*n-1-i))), N, n)
        """cycle through all possible values of the classical register and apply the corresponding conditional phase shiftmpfr(a**(2**(2*n-1-i)))"""
        for j in range(0, 2**i):
            """the phase shift is applied if the value of the classical register matches j exactly"""
            circuit.u1(getAngle(j, i), up_reg[0]).c_if(up_classic, j)
        circuit.h(up_reg)
        circuit.measure(up_reg[0], up_classic[i])
        circuit.measure(up_reg[0], c_aux[0])

    circuit.draw(output='mpl')
    plt.show()


    """ Simulate the created Quantum Circuit """
    simulation = execute(circuit, backend=backend, shots=shots, optimization_level=3, memory=True)
    job_monitor(simulation)
    sim_result = simulation.result()
    counts_result = sim_result.get_counts(circuit)

    if print_statistics:
        dict = {}
        for i in range(256):
            dict[i] = 0
        for i in range(0, len(counts_result)):
            print('Result \"{0}\" ({3}) happened {1} times out of {2}'.format(list(sim_result.get_counts().keys())[i],
                                                                              list(sim_result.get_counts().values())[i],
                                                                              shots,
                                                                              int(list(sim_result.get_counts().keys())[
                                                                                      i].split(" ")[1],
                                                                                  2)), file=stats)
            dict[int(list(sim_result.get_counts().keys())[i].split(" ")[1], 2)] = list(sim_result.get_counts().values())[i]
        print(dict, file=stats)
        for i in range(0, 256, 4):
            for k in range(1, 4):
                dict[i] += int(dict[i + k])
                dict.pop(i + k)
        plot_histogram(dict, figsize=(28, 6), bar_labels=False, title='Measurement for N=15, g=2 (2048 samples)')
        plt.xticks([0, 16, 32, 48, 63], [0, 64, 128, 192, 255])
        plt.show()


    """ Initialize this variable """
    prob_success = 0
    list_of_factors = set()
    """ For each simulation result, print proper info to user and try to calculate the factors of N"""

    for i in range(0, len(counts_result)):

        """ Get the x_value from the final state qubits """
        all_registers_output = list(sim_result.get_counts().keys())[i]
        output_desired = all_registers_output.split(" ")[1]
        x_value = int(output_desired, 2)
        prob_this_result = 100 * (int(list(sim_result.get_counts().values())[i])) / (shots)

        if get_factors(list_of_factors, int(x_value), int(2 * n), int(N), int(a)):
            prob_success = prob_success + prob_this_result

    if print_statistics:
        if len(list_of_factors):
            print("\nUsing a={0}, found the factors {3} for N={1} in {2:.4f} % of the cases\n".format(a,N,prob_success,
                                                                                         list_of_factors), file=stats)
        else:
            print("\nUnable to find factors of N={1} using a={0}\n".format(a, N), file=stats)
        stats.close()

    """In case only one factor was discovered calculate the other"""
    if len(list_of_factors) == 1:
        for a in list_of_factors:
            factor = a
        list_of_factors.add(int(N/factor))
    return list_of_factors