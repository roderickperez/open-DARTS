
from examples.case_1 import input_data_case_1
from examples.case_1_burden import input_data_case_1_burden
from examples.case_1_burden_2 import input_data_case_1_burden_2
from examples.case_2 import input_data_case_2
from examples.case_3 import input_data_case_3
from examples.whitby import input_data_case_whitby

def set_input_data(case: str):
    if case == 'case_1':
        input_data = input_data_case_1()
    elif case == 'case_1_burden':
        input_data = input_data_case_1_burden()
    elif case == 'case_1_burden_2':
        input_data = input_data_case_1_burden_2()
    elif case == 'case_2':
        input_data = input_data_case_2()
    elif case == 'case_3':
        input_data = input_data_case_3()
    elif case == 'whitby':
        input_data = input_data_case_whitby()
    else:
        assert False, f'Wrong case {case}'
    return input_data