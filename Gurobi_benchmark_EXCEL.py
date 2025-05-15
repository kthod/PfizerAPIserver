import numpy as np  # Import NumPy for numerical operations
#import networkx as nx  # Import NetworkX for graph operations (not used in the shown code)
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import matplotlib.colors as mcolors  # Import Matplotlib colors for color operations
#from CompressedVQE_Class import CompressedVQE  # Import the Quantum_MPC class for quantum model predictive control
from VQE_Class import VQE 
from CompressedVQE_Class import CompressedVQE
from QAOA_Class import QAOA 
from Create_QUBO import create_qubo_matrix
from Environment import Environment
from GoogleOR import Google_OR
#from qaoa import MonitoredQAOA
import matplotlib.patches as mpatches
import math
from collections import defaultdict
import pandas as pd

# Load the Excel file
file_path = r"C:\Users\marsi\Downloads\D2\D2\DE22_20240508.xlsx"  # Replace with your Excel file path




#lines_list = [2,5,10, 20,50,100,200,500,800,1000,1500]
num_of_variables = []
num_of_constraints =  []
elapsed_time = []
lines = 30
# Extract the column values as a list
#for lines in lines_list:
df = pd.read_excel(file_path)
p = df['TARGET_PROC_TIME'][:lines].tolist()
Material_key = df['MATERIAL_KEY'][:lines].tolist()
process_order = df['PROCESS_ORDER'][:lines].tolist()
Jobs = df['MATERIAL_KEY'][:lines].tolist()
Brands = df['BRAND_DESC'][:lines].tolist()
mg = df['API1_STRENGTH_KEY'][:lines].tolist()
machines = df['WORK_CENTER_RESOURCE'][:lines].tolist()
#alternative_machines = df['WORK_CENTER_RESOURCE_y'][:lines].tolist()
# Print the resulting array

Jobs = list(zip(Material_key,process_order))




machines_set = set(machines)  

# Map each distinct element to a unique number
machines_to_number = {machine: idx for idx, machine in enumerate(machines_set)}




# Example data
num_jobs = len(set(Jobs))
num_tasks = len(p)
num_machines = len(machines_set)

# Create a dictionary to store task-machine assignments
task_machine_mapping = {}

# Assign each task a random subset of machines it can be executed on
for task in range( num_tasks ):
    #machines_for_task = random.sample(range( num_machines ), random.randint(1, num_machines-1))
    task_machine_mapping[task] = {machines_to_number[machines[task]]}
    #task_machine_mapping[task].update(alternative_machines_dict[(Material_key[task], machines[task])])





lambdas = np.array([1]*1000)
mu = 1
tol=1e-6
n_constraints = len(lambdas)
number_of_experiments = 5
bitstring = '0'
    
env = Environment(num_jobs=num_jobs,
                 num_tasks=num_tasks,
                 num_machines=num_machines,
                 Horizon=32,
                 p = p,
                 Jobs = Jobs,
                 Brands = Brands,
                 mg = mg,
                 task_machine_mapping = task_machine_mapping)  

print(env)
print("Task order pairs:", env.task_order_pairs)
print("Tuples (conflicts):", env.tuples)
print("Task->Bit mapping:", env.task_to_bit)
num_tasks = env.num_tasks
print(num_tasks)
task_to_bit = env.task_to_bit
task_machine_mapping = env.task_machine_mapping
inactivity_window_mapping = env.inactivity_window_mapping
task_order_pairs =env.task_order_pairs
tuples = env.tuples
unique_mapping = env.unique_mapping



Jobs = env.Jobs
p = env.p
u_r = env.u_r
delta = env.delta
delta_star = env.delta_star
t_c = env.t_c
t_c_star = env.t_c_star

n = env.n
M = env.M

num_task_order_constraints = len(task_order_pairs)
num_last_task_constraints = num_tasks
num_overlap_constraints = 2 * len(tuples)    



Q= create_qubo_matrix(env, lambdas, n + num_tasks*(n )+len(tuples))
bitstring_or = Google_OR(env)

inst = VQE(Q, layers=2)

opt_value = inst.compute_expectation({bitstring_or:100})
#opt_value = 0


print(f"opti {opt_value}")

# inst_n1_2 = CompressedVQE(Q, layers=3,na=int(n))

# initial_vector = inst_n1_2.optimize(n_measurements=100000, number_of_experiments=8, maxiter=400)

# inst_n1_2.plot_evolution( color='C5', normalization = [opt_value], label = "CompressedVQE na=n, shots = 100000 ")
# ################################################################################
inst_n = CompressedVQE(Q, layers=3,na=int(n/2))

initial_vector = inst_n.optimize(n_measurements=200000, number_of_experiments=8, maxiter=300)

inst_n.plot_evolution( color='C4', normalization = [opt_value], label = "CompressedVQE na=n/2, shots =300000")
################################################################################
inst_2n = CompressedVQE(Q, layers=3,na=int(n))

initial_vector = inst_2n.optimize(n_measurements=200000, number_of_experiments=8, maxiter=300)

inst_2n.plot_evolution( color='C3', normalization = [opt_value], label = "CompressedVQE na=n, shots = 300000")

#Set up the plot for evaluating the cost function over the optimization iterations
plt.title(f"Evaluation of Cost Function for timesteps", fontsize=16)
plt.ylabel('Cost Function', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.legend(fontsize=12)
plt.show()

bitstring = inst_2n.show_solution(shots = 1000000)
for i in range(num_tasks):
    s = 0
    bit  = task_to_bit[i]
    for q in range(n):
        s+=(2**q) * int(bitstring[bit+q])
    
    #start_time.append(int(bitstring[3+i*4]) + 2*int(bitstring[3+i*4+1]) + 4*int(bitstring[3+i*4+2]))
    print(s)



# Evaluate the constraint violations
first_batch = num_task_order_constraints
second_batch = first_batch + num_last_task_constraints
third_batch = second_batch + num_overlap_constraints
fourth_batch = third_batch + 2*num_tasks
print(f"num of constraints {fourth_batch}")
c_values = []
start_time = []
assigned_machine = []
#print(f"  Current constraint violations: {c_values}")
C_max = 0
for q in range(n):
    C_max += (2**q) * int(bitstring[q])# + 2*int(bitstring[1]) + 4 *int(bitstring[2])
for i in range(num_tasks):
    s = 0
    bit  = task_to_bit[i]
    for q in range(n):
        s+=(2**q) * int(bitstring[bit+q])
    start_time.append(s)
    #start_time.append(int(bitstring[3+i*4]) + 2*int(bitstring[3+i*4+1]) + 4*int(bitstring[3+i*4+2]))
    print(start_time)
    duration = p[i]
    for m in task_machine_mapping[i]:
        if int(bitstring[bit+n]) == 1:
            assigned_machine.append(m) 
# Update Lagrange multipliers
for i in range(num_task_order_constraints):
    # print(f"i {i} machine {assigned_machine[i]} ")
    task_k = task_order_pairs[i][1]
    task_i = task_order_pairs[i][0]
    c_values.append(-(start_time[task_k] - start_time[task_i] - p[task_i]))#*(2 - u_r[assigned_machine[i]])))
    if c_values[i] > 0:  # Only update if the constraint is violated
        lambdas[i] += mu * c_values[i]
        print(f"  Updated lambda task order {task_order_pairs[i]}: {lambdas[i]}")

for i in range(num_last_task_constraints):
    # print(f"i {i} machine {assigned_machine[i]} ")
    j = first_batch +i
    c_values.append(-(C_max + 1 - start_time[i] - p[i]))#*(2 - u_r[assigned_machine[i]])))
    #c_values.append(-1)
    if c_values[j] > 0:  # Only update if the constraint is violated
        lambdas[j] += mu * c_values[j]
        print(f"  Updated lambda[{j}]: {lambdas[j]}")

#tuples = [(0,3),(2,1)]
for i in range(0, num_overlap_constraints, 2):
    print(i)
    
    tuple = tuples[i//2]
    y_ind = unique_mapping[tuple] +task_to_bit[num_tasks-1] + n #+ len(task_machine_mapping[num_tasks-1])+len(inactivity_window_mapping[num_tasks-1])
    
    print(f"len bitstr{len(bitstring)} y_ind {y_ind}")
    y = int(bitstring[y_ind])
    print(f"y {y}")
    c_values.append(-(start_time[tuple[1]] - start_time[tuple[0]] - p[tuple[0]]-delta[tuple[0]][tuple[1]]*t_c - (1-delta[tuple[0]][tuple[1]])*delta_star[tuple[0]][tuple[1]]*t_c_star + M*(1-y)))#*(2 - u_r[assigned_machine[i]])))
    
    j = second_batch+i
    if c_values[j] > 0:  # Only update if the constraint is violated
        lambdas[j] += mu * c_values[j]
        print(f"  Updated lambda no overlap {tuple} {j//2} y=1 [{j}]: {lambdas[j]}")
    
    c_values.append(-(start_time[tuple[0]] - start_time[tuple[1]] - p[tuple[1]]-delta[tuple[0]][tuple[1]]*t_c - (1-delta[tuple[0]][tuple[1]])*delta_star[tuple[0]][tuple[1]]*t_c_star + M*y))#*(2 - u_r[assigned_machine[i]])))
    
    if c_values[j+1] > 0:  # Only update if the constraint is violated
        lambdas[j+1] += mu * c_values[j+1]
        print(f"  Updated lambda no overlap {tuple} {j//2} y=0 [{j+1}]: {lambdas[j+1]}")

for i in range(0, 2*num_tasks, 2):
            
            z_ind = task_to_bit[i//2]+n+len(task_machine_mapping[i//2])

            z = int(bitstring[z_ind])
            
            c_values.append((start_time[i//2] + p[i//2] - inactivity_window_mapping[i//2][0][0]- M*(1-z)))#*(2 - u_r[assigned_machine[i]])))
            j = third_batch + i
            print(f"INSIDE CONSTRIANT LEN {len(c_values)} J {j}")
            if c_values[j] > 0:  # Only update if the constraint is violated
                lambdas[j] += mu * c_values[j]
                print(f"  Updated lambda inacivity window ({i//2}) z=1 [{j}]: {lambdas[j]}")
            
            c_values.append(-(start_time[i//2] - inactivity_window_mapping[i//2][0][1]+ M*z))#*(2 - u_r[assigned_machine[i]])))
            
            if c_values[j+1] > 0:  # Only update if the constraint is violated
                lambdas[j+1] += mu * c_values[j+1]
                print(f"  Updated lambda inacivity window ({i//2}) z=0 [{j+1}]: {lambdas[j+1]}")



# Check stopping criterion: all constraints satisfied
if all(np.array(c_values) <= tol):
    print("Stopping criteria reached: all constraints satisfied.")
    


