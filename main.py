from qiskit import IBMQ, Aer
from Shor import shor4n2, shor2n3

from QuantumSimulations import least_busy_backend
import math
from qiskit.algorithms.factorizers import Shor
IBMQ.save_account('701683288b1d32f91563c19744b8fee55f84b65ce83243c772ac860bc6ede7b8'
                  'ec6364b5087be3e6d81ad3f112224cbdfbc5460aeb8ff5c52c30a821526742c8')

def binary_list(n):
    return ['{:0{}b}'.format(i, n) for i in range(n*n-1)]


if __name__ == '__main__':

    #N = int(input('N: '))
    #a = int(input('a: '))
    N = 15
    a = 2

    IBMQ.load_account()
    provider = IBMQ.get_provider('ibm-q')
    #backend = least_busy_backend(provider, n)
    backend = provider.get_backend('simulator_statevector')
    #backend = Aer.get_backend('qasm_simulator')
    shots = 2048

    factors = shor4n2(N, a, backend=backend, shots=shots, print_statistics=True)

    if factors:
        print(factors)

