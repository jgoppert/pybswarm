#%%
class C1():
    def __init__(self, names):
        self.names = names
        self.funcs = {name: None for name in self.names}

class CF():
    def __init__(self, uri):
        self.uri = uri

def func1():
    print('yeet')

def func2(cf, obj: C1, name):
    obj.funcs[name] = func1

def process_args(cf, func, uri, args_dict):
    args = [cf]
    args += args_dict[uri]
    print(args)
    func(*args)

#%%
names = ['a','b','c']
c = C1(names)

test_args = {name+str(i): [c, name] for i,name in enumerate(names)}
print(test_args)

cf = CF('a0')

process_args(cf, func2, 'a0', test_args)
print('call class')
# c.funcs['a']= func1

for func in c.funcs.values():
    func()
# %%
