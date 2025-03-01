import os
path_to_code = r"/Users/Guille/Desktop/caiso_power/software/"
code_name    = r"CAISO_API_for_local_marginal_prices_new.py"

for i_node in range(100):
    args = r" {}".format(i_node)
    command = "python " + path_to_code + code_name + args
    print(command)
    os.system(command)
