from multiprocessing import freeze_support
from datetime import datetime
from main_gen_mesh import generate_mesh
from main_simulation import run_simulation

def run_test(args: list = []):
    if len(args) > 1:
        return test(case=args[0], overwrite=args[1])
    else:
        print('Not enough arguments provided')
        return 1, 0.0

def test(case, overwrite='0'):
    freeze_support()

    t1 = datetime.now()
    print(t1)

    generate_mesh(case=case)
    run_simulation(case=case)

    t2 = datetime.now()
    print((t2-t1).total_seconds())

    return 0, 0.0

if __name__ == "__main__":
    #cases_list = ['case_1']
    #cases_list = ['case_1_burden']
    #cases_list = ['case_1_burden_2']
    cases_list = ['case_1', 'case_1_burden', 'case_1_burden_2']
    for case in cases_list:
        test(case)