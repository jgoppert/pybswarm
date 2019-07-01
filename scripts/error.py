#%% [markdown]
# # Kalman Filter for UWB TDOA measurement
#
# The time difference of arrival (TDOA) measurement is given as:
# 
# $d = |\vec{a} - \vec{n}| - |\vec{b} - \vec{n}|$
#%%

import sympy
sympy.init_printing()

a = sympy.Matrix(sympy.symbols('a_0:3', real=True))
b = sympy.Matrix(sympy.symbols('b_0:3', real=True))
n = sympy.Matrix(sympy.symbols('n_0:3', real=True))

d = sympy.Matrix.norm(a - n) - sympy.Matrix.norm(b - n)

da = sympy.symbols('da')
db = sympy.symbols('db')


H = sympy.Matrix([d]).jacobian(n)
H.simplify()
H.subs({
    sympy.Matrix.norm(a - n): da,
    sympy.Matrix.norm(b - n): db})


#%%


#%%
