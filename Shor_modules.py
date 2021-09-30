from qiskit import QuantumRegister, QuantumCircuit
from QuantumFourierTransform import qft, inverse_qft
import math
import numpy as np
from gmpy2 import mpfr
import matplotlib.pyplot as plt
from qiskit.circuit.library import MCMTVChain, MCMT

"""n qubit circuit to compute |Φ(a+b)> if is_sub=False"""
"""if is_sub=True computes |Φ(b-a)> for b≥a or |Φ(2^(n-1)-(a-b))> if b<a"""
def phi_add(circuit: QuantumCircuit, a: int, b: QuantumRegister, n: int, is_sub=False):
    angles = calc_angles(a, n)
    qr = QuantumRegister(b.size)

    if is_sub:
        qc = QuantumCircuit(qr, name='phi_add(-)')
        for i in range(0, n):
            qc.p(-angles[i], qr[i])
    else:
        qc = QuantumCircuit(qr, name='phi_add(+)')
        for i in range(0, n):
            qc.p(angles[i], qr[i])

    circuit.append(qc.to_instruction(), qargs=b)


"""Function to calculate angles used in addition in the Fourier Space"""
"""(formula derived in appendix A of the report, eq.A.6)"""
def calc_angles(a: int, n: int) -> np.ndarray:
    angles = np.zeros([n])
    for i in range(0, n):
        angles[i] = (((2 * np.pi * mpfr(a)) / math.pow(2, i + 1)) % (2 * np.pi))
    return angles


"""Single controlled version of the phi_add circuit"""
def cphi_add(circuit: QuantumCircuit, a: int, c1: QuantumRegister, b: QuantumRegister, n: int, is_sub=False):
    angles = calc_angles(a, n)

    qctl = QuantumRegister(1)
    qr = QuantumRegister(n)
    if is_sub:
        qc = QuantumCircuit(qctl, qr, name='cphi_add(-)')
        for i in range(0, n):
            qc.cu1(-angles[i], qctl, qr[i])
    else:
        qc = QuantumCircuit(qctl, qr, name='cphi_add(+)')
        for i in range(0, n):
            qc.cu1(angles[i], qctl, qr[i])

    q_args = np.array(b[:n]).tolist()
    q_args.insert(0, c1)
    circuit.append(qc.to_instruction(), qargs=q_args)


"""Doubly controlled version of phi_add circuit"""
def ccphi_add(circuit: QuantumCircuit, a: int, c1:QuantumRegister, c2:QuantumRegister, b: QuantumRegister, n: int, is_sub=False):
    angles = calc_angles(a, n)

    qctl = QuantumRegister(2)
    qr = QuantumRegister(n)

    if is_sub:
        qc = QuantumCircuit(qctl, qr, name='ccphi_add(-)')
        for i in range(0, n):
            ccphase(qc, -angles[i], qctl[0], qctl[1], qr[i])
    else:
        qc = QuantumCircuit(qctl, qr, name='ccphi_add(+)')
        for i in range(0, n):
            ccphase(qc, angles[i], qctl[0], qctl[1], qr[i])

    q_args = np.array(b[:n]).tolist()
    q_args.insert(0, c2)
    q_args.insert(0, c1)
    circuit.append(qc.to_instruction(), qargs=q_args)


"""Creation of a doubly controlled phase gate"""
def ccphase(qc:QuantumCircuit, angle: float, c1:QuantumRegister, c2:QuantumRegister, qr:QuantumRegister):

    qc.cu1(angle/2, c1, qr)
    qc.cx(c2, c1)
    qc.cu1(-angle/2, c1, qr)
    qc.cx(c2, c1)
    qc.cu1(angle/2, c2, qr)


def ccphi_add_mod_N(circuit: QuantumCircuit, a: int, c1: QuantumRegister, c2: QuantumRegister, b: QuantumRegister, n: int, N: int, return_inverse=False):

    qctl = QuantumRegister(2)
    qr = QuantumRegister(n)
    qaux = QuantumRegister(1)
    qc = QuantumCircuit(qctl, qr, qaux, name='ccphi_add_modN')

    ccphi_add(qc, a, qctl[0], qctl[1], qr, n)
    phi_add(qc, N, qr, n, is_sub=True)
    qc.append(inverse_qft(n, swap=0), qr)
    qc.cx(qr[n-1], qaux)
    qc.append(qft(n, swap=0), qr)
    cphi_add(qc, N, qaux, qr, n)

    ccphi_add(qc, a, qctl[0], qctl[1], qr, n, is_sub=True)
    qc.append(inverse_qft(n, swap=0), qr)
    qc.x(qr[n-1])
    qc.cx(qr[n-1],qaux)
    qc.x(qr[n-1])
    qc.append(qft(n, swap=0), qr)
    ccphi_add(qc, a, qctl[0], qctl[1], qr, n)

    q_args = np.array(b).tolist()
    q_args.insert(0, c2)
    q_args.insert(0, c1)
    circuit.append(qc.to_instruction(), qargs=q_args)

    if return_inverse:
        return qc.inverse()
    else:
        return


def cmult_mod_N(circuit: QuantumCircuit, c: QuantumRegister, x: QuantumRegister, b: QuantumRegister, a, N: int, n:int):

    qc = QuantumRegister(1)
    qx = QuantumRegister(x.size)
    qb = QuantumRegister(b.size)
    qcirc = QuantumCircuit(qc, qx, qb, name='U_a')

    qcirc.append(qft(b.size - 1, swap=0), qargs=qb[:b.size - 1])

    # Save inverse of cmult_mod_N gates for future use
    invcmult_list = []

    for i in range(0, n):
        aux_ext = mpfr(math.pow(2, i) * a) % N
        invcmult_list.insert(0, ccphi_add_mod_N(qcirc, aux_ext, qx[i], qc, qb, n + 1, N, return_inverse=True))

    qcirc.append(inverse_qft(b.size - 1, swap=0), qargs=qb[:b.size - 1])

    for i in range(0, n):
        qcirc.cswap(qc, qx[i], qb[i])

    qcirc.append(qft(b.size - 1, swap=0), qargs=qb[:b.size - 1])

    i = n - 1
    for inv_circ in invcmult_list:
        q_args = np.array(qb).tolist()
        q_args.insert(0, qc)
        q_args.insert(0, qx[i])
        qcirc.append(inv_circ, qargs=q_args)
        i -= 1

    qcirc.append(inverse_qft(b.size - 1, swap=0), qargs=qb[:b.size - 1])

    q_args = np.array(x).tolist() + np.array(b).tolist()
    q_args.insert(0, c)
    circuit.append(qcirc.to_instruction(), qargs=q_args)

def cphi_add_mod_N(circuit: QuantumCircuit, a: int, c1: QuantumRegister, b: QuantumRegister, n: int, N: int, return_inverse=False):

    qctl = QuantumRegister(1)
    qr = QuantumRegister(n)
    qaux = QuantumRegister(1)
    qc = QuantumCircuit(qctl, qr, qaux, name='cphi_add_modN')

    cphi_add(qc, a, qctl[0], qr, n)
    phi_add(qc, N, qr, n, is_sub=True)
    qc.append(inverse_qft(n, swap=0), qr)
    qc.cx(qr[n-1], qaux)
    qc.append(qft(n, swap=0), qr)
    cphi_add(qc, N, qaux, qr, n)

    cphi_add(qc, a, qctl[0], qr, n, is_sub=True)
    qc.append(inverse_qft(n, swap=0), qr)
    qc.x(qr[n-1])
    qc.cx(qr[n-1],qaux)
    qc.x(qr[n-1])
    qc.append(qft(n, swap=0), qr)
    cphi_add(qc, a, qctl[0], qr, n)

    q_args = np.array(b).tolist()
    q_args.insert(0, c1)
    circuit.append(qc.to_instruction(), qargs=q_args)

    if return_inverse:
        return qc.inverse()
    else:
        return


def c_adder(circuit: QuantumCircuit, a: int, c: QuantumRegister, b: QuantumRegister, n: int, is_sub=False):
    aux_reg = QuantumRegister(n)
    ctrl_reg = QuantumRegister(1)
    qc = QuantumCircuit(ctrl_reg, aux_reg)
    qc.append(qft(n, swap=0), aux_reg)
    cphi_add(qc, a, ctrl_reg, aux_reg, n, is_sub)
    qc.append(inverse_qft(n, swap=0), aux_reg)
    q_args = np.array(b).tolist()
    q_args.insert(0, c)
    circuit.append(qc.to_instruction(), qargs=q_args)

def adder(circuit: QuantumCircuit, a: int, b: QuantumRegister, n: int, is_sub=False):
    aux_reg = QuantumRegister(n)
    qc = QuantumCircuit(aux_reg)
    qc.append(qft(n, swap=0), aux_reg)
    phi_add(qc, a, aux_reg, n, is_sub)
    qc.append(inverse_qft(n, swap=0), aux_reg)
    q_args = np.array(b).tolist()
    circuit.append(qc.to_instruction(), qargs=q_args)

"""B+A or B-A"""
def two_reg_adder(circuit: QuantumCircuit, a: QuantumRegister, b: QuantumRegister, n: int, is_sub=False):

    qa = QuantumRegister(n)
    qb = QuantumRegister(n)
    qc = QuantumCircuit(qa, qb)

    for i in range(0, n-1):
        qc.cx(qa[n-1], qa[i])
        qc.cx(qa[n-1], qb[i])
    qc.cx(qa[n-1], qb[n-1])

    for i in range(0, n-1):
        qc.cx(qa[n-1], qb[i])
        qc.cswap(qb[i],qa[n-1], qa[i])
    qc.cx(qa[n-1], qb[n-1])

    for i in range(n-2, -1, -1):
        qc.cswap(qb[i],qa[n-1], qa[i])
        qc.cx(qa[i], qb[i])

    for i in range(0, n-1):
        qc.cx(qa[n-1], qa[i])
        qc.cx(qa[n-1], qb[i])
    qc.cx(qa[n-1], qb[n-1])

    q_args = np.array(a).tolist()+np.array(b).tolist()

    if is_sub:
        circuit.append(qc.inverse().to_instruction(), qargs=q_args)
    else:
        circuit.append(qc.to_instruction(), qargs=q_args)


def controlled_increment(circuit: QuantumCircuit, target: QuantumRegister, dirty: QuantumRegister,
                         ctrl: QuantumRegister, n: int, is_sub=False):

    """IF N IS EVEN"""
    if (n % 2) == 0:
        reg_size = math.ceil(n / 2)
        qctrl = QuantumRegister(1)
        lsb_control = QuantumRegister(1)
        upper = QuantumRegister(reg_size)
        lower = QuantumRegister(reg_size)
        qc = QuantumCircuit(qctrl, lsb_control, upper, lower)

        multi_control_x_gate = MCMT('x', num_ctrl_qubits=reg_size + 2, num_target_qubits=reg_size)
        single_control_x_gate = MCMT('x', num_ctrl_qubits=2, num_target_qubits=reg_size * 2)

        for q in range(reg_size // 2):
            qc.swap(lower[q], lower[reg_size - q - 1])

        two_reg_adder(qc, upper, lower, reg_size, is_sub=True)
        q_args = np.array(upper).tolist() + np.array(lower).tolist()
        q_args.insert(0, lsb_control)
        q_args.insert(0, qctrl)
        qc.append(multi_control_x_gate.to_instruction(), qargs=q_args)
        two_reg_adder(qc, upper, lower, reg_size, is_sub=False)
        qc.append(multi_control_x_gate.to_instruction(), qargs=q_args)
        two_reg_adder(qc, lower, upper, reg_size, is_sub=True)
        qc.append(single_control_x_gate.to_instruction(), qargs=q_args)
        two_reg_adder(qc, lower, upper, reg_size, is_sub=False)
        qc.append(single_control_x_gate.to_instruction(), qargs=q_args)

        qc.x(lsb_control)
        for q in range(reg_size // 2):
            qc.swap(lower[q], lower[reg_size - q - 1])

        q_args = np.array(target).tolist()
        q_args.append(dirty)
        q_args.insert(0, ctrl)
        if is_sub:
            circuit.append(qc.inverse().to_instruction(), qargs=q_args)
        else:
            circuit.append(qc.to_instruction(), qargs=q_args)
    else:
        """N IS ODD"""
        reg_size = math.ceil(n / 2)
        qctrl = QuantumRegister(1)
        upper = QuantumRegister(reg_size)
        lower = QuantumRegister(reg_size)
        qc = QuantumCircuit(qctrl, upper, lower)

        multi_control_x_gate = MCMT('x', num_ctrl_qubits=reg_size + 1, num_target_qubits=reg_size)
        single_control_x_gate = MCMT('x', num_ctrl_qubits=1, num_target_qubits=reg_size * 2)

        for q in range(reg_size // 2):
            qc.swap(lower[q], lower[reg_size - q - 1])

        two_reg_adder(qc, upper, lower, reg_size, is_sub=True)
        q_args = np.array(upper).tolist() + np.array(lower).tolist()
        q_args.insert(0, qctrl)
        qc.append(multi_control_x_gate.to_instruction(), qargs=q_args)
        two_reg_adder(qc, upper, lower, reg_size, is_sub=False)
        qc.append(multi_control_x_gate.to_instruction(), qargs=q_args)
        two_reg_adder(qc, lower, upper, reg_size, is_sub=True)
        qc.append(single_control_x_gate.to_instruction(), qargs=q_args)
        two_reg_adder(qc, lower, upper, reg_size, is_sub=False)
        qc.append(single_control_x_gate.to_instruction(), qargs=q_args)

        for q in range(reg_size // 2):
            qc.swap(lower[q], lower[reg_size - q - 1])

        q_args = np.array(target).tolist()
        q_args.append(dirty)
        q_args.insert(0, ctrl)
        if is_sub:
            circuit.append(qc.inverse().to_instruction(), qargs=q_args)
        else:
            circuit.append(qc.to_instruction(), qargs=q_args)

def increment(circuit: QuantumCircuit, target: QuantumRegister, dirty: QuantumRegister, n: int):

    qtarget = QuantumRegister(n)
    qdirty = QuantumRegister(n)
    qc = QuantumCircuit(qtarget, qdirty)

    two_reg_adder(qc, qdirty, qtarget, n, is_sub=True)
    for i in range(0, n):
        qc.x(qdirty[i])
    two_reg_adder(qc, qdirty, qtarget, n, is_sub=True)
    for i in range(0, n):
        qc.x(qdirty[i])

    q_args = np.array(target).tolist()+np.array(dirty).tolist()
    circuit.append(qc.to_instruction(), qargs=q_args)

"""B+A or B-A with target larget than source"""
def two_reg_adder_large_target(circuit: QuantumCircuit, a: QuantumRegister, b: QuantumRegister, n: int, is_sub=False):

    n_a = a.size
    if type(b) is list:
        n_b = len(b)
    else:
        n_b = b.size
    qa = QuantumRegister(n_a)
    qb = QuantumRegister(n_b)
    qc = QuantumCircuit(qa, qb)
    e = n_b-n_a

    controlled_increment(qc, qb, qa[0], qa[n_a-1], n_b, is_sub=True)
    controlled_increment(qc, qb[n_b-e-1:n_b], qa[0], qa[n_a - 1], e+1, is_sub=True)

    for i in range(0, n_a-1):
        qc.cx(qa[n_a-1], qb[i])
        qc.cswap(qb[i], qa[n-1], qa[i])

    controlled_increment(qc, qb[n_b-e-1:n_b], qa[0], qa[n_a - 1], e+1, is_sub=True)

    for i in range(n_a-2, -1, -1):
        qc.cswap(qb[i], qa[n-1], qa[i])
        qc.cx(qa[i], qb[i])

    q_args = np.array(a).tolist()+np.array(b).tolist()

    if is_sub:
        circuit.append(qc.inverse().to_instruction(), qargs=q_args)
    else:
        circuit.append(qc.to_instruction(), qargs=q_args)

def carry(circuit: QuantumCircuit, a: QuantumRegister, dirty: QuantumRegister,
          zero: QuantumRegister, const: str, n: int):

    qa = QuantumRegister(n)
    qd = QuantumRegister(n-1)
    qz = QuantumRegister(1)
    qc = QuantumCircuit(qa, qd, qz)

    qc.cx(qd[n-2], qz)
    carry_rec(qc, qa, qd, const, n-1)
    qc.cx(qd[n-2], qz)
    carry_rec_rev(qc, qa, qd, const, n-1)

    q_args = np.array(a).tolist()+np.array(dirty).tolist()
    q_args.append(zero)
    circuit.append(qc.to_instruction(), qargs=q_args)


def carry_rec(circuit: QuantumCircuit, a: QuantumRegister, dirty: QuantumRegister,
          c: str, iter: int):

    if c[iter] == '1':
        circuit.cx(a[iter], dirty[iter-1])
        circuit.x(a[iter])

    if iter == 1:
        if c[iter-1] == '1':
            circuit.ccx(a[iter], a[iter - 1], dirty[iter - 1])
        return

    circuit.ccx(a[iter], dirty[iter-2], dirty[iter-1])
    carry_rec(circuit, a, dirty, c, iter-1)
    circuit.ccx(a[iter], dirty[iter - 2], dirty[iter - 1])



def carry_rec_rev(circuit: QuantumCircuit, a: QuantumRegister, dirty: QuantumRegister,
              c: str, iter: int):

    if iter == 1:
        if c[iter-1] == '1':
            circuit.ccx(a[iter], a[iter - 1], dirty[iter-1])
            circuit.x(a[iter])
            circuit.cx(a[iter], dirty[iter-1])
        return

    circuit.ccx(a[iter], dirty[iter - 2], dirty[iter-1])
    carry_rec_rev(circuit, a, dirty, c, iter-1)
    circuit.ccx(a[iter], dirty[iter - 2], dirty[iter-1])

    if c[iter] == '1':
        circuit.x(a[iter])
        circuit.cx(a[iter], dirty[iter-1])


def offset(circuit: QuantumCircuit, a: QuantumRegister, b: QuantumRegister,
              dirty: QuantumRegister, c: str):

    qa = QuantumRegister(len(a))
    qb = QuantumRegister(len(b))
    qd = QuantumRegister(dirty.size)
    qc = QuantumCircuit(qa, qb, qd)
    kl = c[:len(a)]
    kh = c[len(b):]

    single_control_x_gate = MCMT('x', num_ctrl_qubits=1, num_target_qubits=len(b))
    q_args = np.array(qb).tolist()
    q_args.insert(0, qd)

    controlled_increment(qc, qb, qa[0], qd, qb.size, is_sub=False)
    qc.append(single_control_x_gate.to_instruction(), qargs=q_args)
    carry(qc, qa, qb[:len(a)-1], qd, c, len(a))
    controlled_increment(qc, qb, qa[0], qd, qb.size, is_sub=False)
    carry(qc, qa, qb[:len(a)-1], qd, c, len(a))
    qc.append(single_control_x_gate.to_instruction(), qargs=q_args)

    if len(kl) > 2:
        offset(qc, qa[:math.floor(len(a)/2)], qa[math.floor(len(a)/2):], qd, kl)
    else:
        if kl[0] == '1':
            qc.x(qa[0])
        if kl[1] == '1':
            qc.x(qa[1])

    if len(kh) > 2:
        offset(qc, qb[:math.floor(len(b)/2)], qb[math.floor(len(b)/2):], qd, kh)
    else:
        if kh[0] == '1':
            qc.x(qb[0])
        if kh[1] == '1':
            qc.x(qb[1])

    q_args = np.array(a).tolist() + np.array(b).tolist()
    q_args.append(dirty)
    circuit.append(qc.to_instruction(), qargs=q_args)


def c_const_compare(circuit: QuantumCircuit, a: QuantumRegister, b: int, c: QuantumRegister, target: QuantumRegister, n:int):

    qinput = QuantumRegister(n+1)
    qctrl = QuantumRegister(1)
    qc = QuantumCircuit(qctrl, qinput)

    c_adder(qc, b, qctrl, qinput, n+1, is_sub=True)
    c_adder(qc, b, qctrl, qinput[:n], n, is_sub=False)

    q_args = np.array(a).tolist()
    q_args.append(target)
    q_args.insert(0, c)
    circuit.append(qc.to_instruction(), qargs=q_args)

def const_compare(circuit: QuantumCircuit, a: QuantumRegister, b: int, target: QuantumRegister, n:int):

    qinput = QuantumRegister(n+1)
    qc = QuantumCircuit(qinput)

    adder(qc, b, qinput, n+1, is_sub=True)
    adder(qc, b, qinput[:n], n, is_sub=False)

    q_args = np.array(a).tolist()
    q_args.append(target)
    circuit.append(qc.to_instruction(), qargs=q_args)

def reg_compare(circuit: QuantumCircuit, a: QuantumRegister, b: QuantumRegister, target: QuantumRegister, n:int):

    qa = QuantumRegister(n)
    qb = QuantumRegister(n+1)
    qc = QuantumCircuit(qa, qb)

    two_reg_adder_large_target(qc, qa, qb, n, is_sub=True)
    #two_reg_adder(qc, qa, qb[:n], n, is_sub=False)

    q_args = np.array(a).tolist()+np.array(b).tolist()
    q_args.append(target)
    circuit.append(qc.to_instruction(), qargs=q_args)

"""B+A or B-A with target larget than source"""
def test_two_reg_adder_large_target(circuit: QuantumCircuit, a: QuantumRegister, b: QuantumRegister, n: int, is_sub=False):

    n_a = a.size
    if type(b) is list:
        n_b = len(b)
    else:
        n_b = b.size
    qa = QuantumRegister(n_a)
    qb = QuantumRegister(n_b)
    qc = QuantumCircuit(qa, qb)
    e = n_b-n_a

    if is_sub:
        for i in range(0, n_a):
            qc.x(qa[i])
        increment(qc, qa, qb[:n_a], n)

    controlled_increment(qc, qb, qa[0], qa[n_a-1], n_b, is_sub=True)
    controlled_increment(qc, qb[n_b-e-1:n_b], qa[0], qa[n_a - 1], e+1, is_sub=True)

    for i in range(0, n_a-1):
        qc.cx(qa[n_a-1], qb[i])
        qc.cswap(qb[i], qa[n-1], qa[i])

    controlled_increment(qc, qb[n_b-e-1:n_b], qa[0], qa[n_a - 1], e+1, is_sub=True)

    for i in range(n_a-2, -1, -1):
        qc.cswap(qb[i], qa[n-1], qa[i])
        qc.cx(qa[i], qb[i])

    q_args = np.array(a).tolist()+np.array(b).tolist()

    if is_sub:
        for i in range(0, n_a):
            qc.x(qa[i])
        increment(qc, qa, qb[:n_a], n)
    circuit.append(qc.to_instruction(), qargs=q_args)