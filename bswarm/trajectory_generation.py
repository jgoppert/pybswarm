import numpy as np
import matplotlib.pyplot as plt


class Trajectory:

    def __init__(self, c, T, n):
        self.c = c
        self.T = T
        self.n = n
        assert len(c) == len(T) * self.n

    @property
    def n_legs(self):
        return len(self.T)

    @property
    def n_coeff(self):
        return self.n * self.n_legs

    def compute_trajectory(self):
        t = []
        x = []
        v = []
        a = []
        j = []
        s = []
        c = self.c
        T = self.T
        n = self.n
        S = np.hstack([[0], np.cumsum(T)])
        for i in range(self.n_legs):

            ti = np.linspace(S[i], S[i + 1])
            beta = compute_beta(ti, S[i], T[i])
            cp = c[n * i:n * (i + 1)]
            cv = poly_derivative(cp, T[i])
            ca = poly_derivative(cv, T[i])
            cj = poly_derivative(ca, T[i])
            cs = poly_derivative(cj, T[i])

            xi = np.polyval(cp, beta)
            vi = np.polyval(cv, beta)
            ai = np.polyval(ca, beta)
            ji = np.polyval(cj, beta)
            si = np.polyval(cs, beta)

            t.append(ti)
            x.append(xi)
            v.append(vi)
            a.append(ai)
            j.append(ji)
            s.append(si)

        t = np.hstack(t)
        x = np.hstack(x)
        v = np.hstack(v)
        a = np.hstack(a)
        j = np.hstack(j)
        s = np.hstack(s)
        return {
            't': t,
            'x': x,
            'v': v,
            'a': a,
            'j': j,
            's': s
        }

    def plot(self):
        res = self.compute_trajectory()
        plt.subplot(511)
        plt.plot(res['t'], res['x'], label='x')
        plt.ylabel('x')
        plt.subplot(512)
        plt.plot(res['t'], res['v'], label='v')
        plt.ylabel('v')
        plt.subplot(513)
        plt.plot(res['t'], res['a'], label='a')
        plt.ylabel('a')
        plt.subplot(514)
        plt.plot(res['t'], res['j'], label='j')
        plt.ylabel('j')
        plt.subplot(515)
        plt.plot(res['t'], res['s'], label='s')
        plt.ylabel('s')
        plt.xlabel('t, sec')


def compute_beta(t, t0, T):
    """
    @param t: elapsed time
    @param t0: initial time for start of leg
    @param T: duration for leg
    """
    return (t - t0) / T


def trajectory_coeff(beta, m, n, T):
    """
    @param beta: scaled time
    @param m: derivative order, 0, for no derivative
    @param n: number of coeff in polynomial (order + 1)
    @param T: period
    """
    p = np.zeros(n)
    for k in range(m, n):
        p[n - k - 1] = (np.math.factorial(k) * beta**(k - m)) / \
            ((np.math.factorial(k - m)) * T**m)
    return p


def poly_derivative(coef, Ti):
    c = []
    n = len(coef)
    for i in range(n - 1):
        ci = (n - 1 - i) * coef[i] / Ti
        c.append(ci)
    c = np.array(c)
    return c


def solve_trajectory(A: np.array, b: np.array, T: list, n: int):
    assert A.shape[0] == b.shape[0]
    assert A.shape[1] == len(T) * n

    rank = np.linalg.matrix_rank(A)
    n_coeff = A.shape[0]
    if rank < n_coeff:
        print('Matrix A not full rank, check constraints, rank: ', rank, '/', n_coeff)
        c = np.linalg.pinv(A).dot(b)
    else:
        c = np.linalg.inv(A).dot(b)
    return Trajectory(c, T, n)


def plan_min_accel(x_list, T):
    """
    @param x_list: position list
    @param T: duration list, length one less than x/v list
    """
    n = 4  # this will be fixed to support 3rd order polynomials
    if not len(x_list) == len(T) + 1:
        raise ValueError('x_list must be 1 longer than T')

    n_legs = len(T)
    n_coeff = n_legs * n
    A = np.zeros((n_coeff, n_coeff))
    b = np.zeros(n_coeff)

    eq_num = 0
    for i in range(n_legs):
        # p1(0) = x[0]
        A[eq_num, n * i:n * (i + 1)] = trajectory_coeff(
            beta=0, m=0, n=n, T=T[i])
        b[eq_num] = x_list[i]
        eq_num += 1

        # initial conitions
        if i == 0:
            for m in [1]:
                # p0^m(0) = 0
                A[eq_num, n * i:n * (i + 1)] = trajectory_coeff(
                    beta=0, m=m, n=n, T=T[i])
                b[eq_num] = 0
                eq_num += 1 
        
        # final conditions
        if i == n_legs - 1:
            for m in [0, 1]:
                # p(n-1)^m(1) = 0
                A[eq_num, n * i:n * (i + 1)] = trajectory_coeff(
                    beta=1, m=m, n=n, T=T[i])
                if m == 0:
                    b[eq_num] = x_list[i+1]
                else:
                    b[eq_num] = 0
                eq_num += 1

        # continuity
        if i > 0:
            # p(i-1)^m(1) - p(i)^m(0) = 0
            for m in [0, 1, 2]:
                A[eq_num, n * (i-1):n * i] = trajectory_coeff(
                    beta=1, m=m, n=n, T=T[i-1])
                A[eq_num, n * i:n * (i + 1)] = -trajectory_coeff(
                    beta=0, m=m, n=n, T=T[i])
                b[eq_num] = 0
                eq_num += 1     

    return solve_trajectory(A, b, T, n)


def plan_min_snap(x_list, T):
    """
    @param x_list: position list
    @param T: duration list, length one less than x/v list
    """
    n = 8  # this will be fixed to support 3rd order polynomials
    if not len(x_list) == len(T) + 1:
        raise ValueError('x_list must be 1 longer than T')

    n_legs = len(T)
    n_coeff = n_legs * n
    A = np.zeros((n_coeff, n_coeff))
    b = np.zeros(n_coeff)

    eq_num = 0
    for i in range(n_legs):

        # conditions
        for m in [0, 1, 2]:
            # p0^(m)(0) = 0
            A[eq_num, n * i:n * (i + 1)] = trajectory_coeff(
                beta=0, m=m, n=n, T=T[i])
            if m == 0:
                b[eq_num] = x_list[i]
            else:
                b[eq_num] = 0
            eq_num += 1

        # initial conditions
        if i == 0:
            # p(0)^4(0) = 0
            A[eq_num, n * i:n * (i + 1)] = trajectory_coeff(
                beta=0, m=4, n=n, T=T[i])
            if m == 0:
                b[eq_num] = x_list[i+1]
            else:
                b[eq_num] = 0
            eq_num += 1

        # final conditions
        if i == n_legs - 1:
            for m in [0, 1, 2, 3]:
                # p(n-1)^m(1) = 0
                A[eq_num, n * i:n * (i + 1)] = trajectory_coeff(
                    beta=1, m=m, n=n, T=T[i])
                if m == 0:
                    b[eq_num] = x_list[i+1]
                else:
                    b[eq_num] = 0
                eq_num += 1

        # continuity
        if i > 0:
            # p(i-1)^m(1) - p(i)^m(0) = 0
            for m in [0, 1, 2, 3, 4]:
                A[eq_num, n * (i-1):n * i] = trajectory_coeff(
                    beta=1, m=m, n=n, T=T[i-1])
                A[eq_num, n * i:n * (i + 1)] = -trajectory_coeff(
                    beta=0, m=m, n=n, T=T[i])
                b[eq_num] = 0
                eq_num += 1     

    return solve_trajectory(A, b, T, n)
