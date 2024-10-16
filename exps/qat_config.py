model_name = 'ResNet'
qbits = 4
qtype = 'float'
format = 'e2m1'
qblock_size = None

save_name = 'exps/results/' + 'qat_' +  qtype + '_' + str(qbits) + '_' + str(qblock_size) + ('_' + format if qtype=='float' else '') + '_' + model_name + '.pickle'

qconfigs = [
    {
        'label': "FP32",
        'activation': {'qtype': None},
        'weight': {'qtype': None},
        'grad': {'qtype': None}
    },
    {
        'label': "FP4-minmax",
        'activation': {'qtype': qtype, 'qbits': qbits, 'qblock_size': qblock_size, 'alg': 'minmax', 'beta':0., 'format': format},
        'weight': {'qtype': qtype, 'qbits': qbits, 'qblock_size': qblock_size, 'alg': 'minmax', 'beta':0., 'format': format},
        'grad': {'qtype': None, 'qbits': qbits, 'qblock_size': qblock_size, 'alg': 'minmax', 'beta':0., 'format': format}
    },
    {
        'label': "FP4-analytic",
        'activation': {'qtype': qtype, 'qbits': qbits, 'qblock_size': qblock_size, 'alg': 'iterative', 'beta':0., 'format': format},
        'weight': {'qtype': qtype, 'qbits': qbits, 'qblock_size': qblock_size, 'alg': 'normal', 'beta':0., 'format': format},
        'grad': {'qtype': None, 'qbits': qbits, 'qblock_size': qblock_size, 'alg': 'normal', 'beta':0., 'format': format}
    },
    {
        'label': "FP4-iterative",
        'activation': {'qtype': qtype, 'qbits': qbits, 'qblock_size': qblock_size, 'alg': 'iterative', 'beta':0., 'format': format},
        'weight': {'qtype': qtype, 'qbits': qbits, 'qblock_size': qblock_size, 'alg': 'iterative', 'beta':0., 'format': format},
        'grad': {'qtype': None, 'qbits': qbits, 'qblock_size': qblock_size, 'alg': 'iterative', 'beta':0., 'format': format}
    }
    ]

