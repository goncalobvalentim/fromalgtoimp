from numpy import pi
from qiskit import QuantumCircuit, QuantumRegister


# Receive a circuit and apply QFT to target register
def qft(n, swap = 1):

    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    if swap == 1:
        qc.name = 'QFT'
    else:
        qc.name = 'QFT\n(no swap)'

    make_subcircuit(qc, qr, n)

    # swap to make last qubit the least significant one
    if swap:
        for q in range(n//2):
            qc.swap(qr[q], qr[n-q-1])

    return qc


def make_subcircuit(circuit, target_reg, n):

    if n == 0:
        return circuit
    n -= 1
    circuit.h(target_reg[n])
    for qubit in range(n):
        circuit.cu1(pi/2**(n-qubit), target_reg[qubit], target_reg[n])

    make_subcircuit(circuit, target_reg, n)


def inverse_qft(n, swap = 1):

    qft_circ = qft(n=n, swap=swap)
    # Then we take the inverse of this circuit
    invqft_circ = qft_circ.inverse()
    if swap == 1:
        invqft_circ.name = 'inverseQFT'
    else:
        invqft_circ.name = 'inverseQFT\n(no swap)'
    return invqft_circ




