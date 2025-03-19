
class qconfig():
    def __init__(self,
                 qtype = None,
                 qbits = 4,
                 qblock_size = None,
                 alg = 'minmax',
                 beta = 0.,
                 format = 'e2m1',
                 ):
        
        self.qtype = qtype
        self.qbits = qbits
        self.qblock_size = qblock_size
        self.alg = alg
        self.beta = beta
        self.format = format


class rconfig():
    def __init__(self,
                 label,
                 activation,
                 weight,
                 grad,
                 ):
        
        self.label = label
        self.activation = activation
        self.weight = weight
        self.grad = grad