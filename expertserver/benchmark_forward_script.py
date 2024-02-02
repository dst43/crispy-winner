
from argparse import ArgumentParser
from itertools import product

def main():
    parser = ArgumentParser()
    args = parser.parse_args()

    file = "benchmark_forward.py"

    general_args = [
        "--epochs=100",
    ]

    param_dict = {
        'embed-dim': [256, 512, 1024, 2048],
        'batch-size': [1, 2, 4, 8],
        'fp16': [True, False],
    }

    products = [list(zip(param_dict, v)) for v in product(*param_dict.values())]
    
    print(products[0])
    
    # cmdlines = []
    # def summarize(s) -> str:
    #     return ''.join([(t[0] if len(t) > 0 else '') for t in str(s).split('-')]).replace('_','')

    # for extra_args in products:
    #     key = '_'.join([summarize(k) + str(v).replace('_','') for k, v in extra_args if k != 'batch-size'])
    #     cmt = '_'.join([summarize(k) + str(v).replace('_','') for k, v in extra_args])
        
    #     cmdlines.append(" ".join([
    #         "python",
    #         file, 
    #         *general_args,
    #         *[f"--{k}={v}" for k, v in extra_args],
    #         f"--key={key}",
    #         f"--comment={cmt}",
    #     ]))
    
    # for cmdline in cmdlines:
    #     print(cmdline)

if __name__ == "__main__":
    main()
