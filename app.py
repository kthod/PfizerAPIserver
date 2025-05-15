from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Any
import uvicorn
import logging
from CompressedVQE_Class import CompressedVQE
from Create_QUBO import create_qubo_matrix
from Environment import Environment
from GoogleOR import Google_OR
import qiskit
#from qiskit.utils import QuantumInstance
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import JSONResponse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce Qiskit's logging verbosity
qiskit_logger = logging.getLogger('qiskit')
qiskit_logger.setLevel(logging.WARNING)

app = FastAPI(
    title="Quantum Optimization API",
    description="API for quantum optimization of scheduling problems",
    version="1.0.0"
)

class OptimizationRequest(BaseModel):
    num_jobs: int
    num_tasks: int
    num_machines: int
    p: List[float]
    Jobs: List[tuple]
    Brands: List[str]
    mg: List[str]
    task_machine_mapping: Dict[str, List[int]]
    #int_to_machine: Dict[int, str]
    layers: int = 3
    n_measurements: int = 20000
    number_of_experiments: int = 1
    maxiter: int = 300

class OptimizationResponse(BaseModel):
    solution: str  # The bitstring solution
    cost_function_value: float
    cost_evolution: Dict[str, List[float]] = {  # Dictionary containing upper, lower, and mean bounds
        "upper_bound": [],
        "lower_bound": [],
        "mean": []
    }

def create_optimization_plot(cost_history: List[float], start_times: List[int], assigned_machines: List[int]) -> Dict[str, Any]:
    """Create plot data for the optimization results."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot cost function evolution
    if cost_history:
        ax1.plot(cost_history)
        ax1.set_title('Cost Function Evolution')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.grid(True)
    
    # Plot Gantt chart
    tasks = range(len(start_times))
    ax2.barh(tasks, [1] * len(tasks), left=start_times, height=0.5)
    ax2.set_title('Task Schedule')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Task')
    ax2.grid(True)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Convert plot to base64 string
    plot_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Create plot data for client-side plotting
    plot_data = {
        'cost_history': cost_history,
        'start_times': start_times,
        'assigned_machines': assigned_machines,
        'tasks': list(tasks)
    }
    
    plt.close()
    
    return {
        'plot_data': plot_data,
        'plot_image': plot_image
    }

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "message": "Welcome to the Quantum Optimization API",
        "documentation": "/docs",
        "endpoints": {
            "GET /": "This information page",
            "POST /optimize": "Submit optimization problem",
            "GET /docs": "Interactive API documentation"
        }
    }

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize(request: OptimizationRequest):
    """Optimize a scheduling problem using quantum computing."""
    try:
        logger.info("Received optimization request")
        logger.info(f"Number of tasks: {request.num_tasks}")
        logger.info(f"Number of machines: {request.num_machines}")
        logger.info(f"Number of jobs: {request.num_jobs}")
        
        # Log the first few entries of each list for debugging
        logger.info(f"First 3 processing times: {request.p[:3]}")
        logger.info(f"First 3 jobs: {request.Jobs[:3]}")
        logger.info(f"First 3 brands: {request.Brands[:3]}")
        logger.info(f"First 3 machine groups: {request.mg[:3]}")
        logger.info(f"First 3 task-machine mappings: {dict(list(request.task_machine_mapping.items())[:3])}")

        # Create environment
        logger.info("Creating environment...")
        try:
            env = Environment(
                num_jobs=request.num_jobs,
                num_tasks=request.num_tasks,
                num_machines=request.num_machines,
                Horizon=32,
                p=request.p,
                Jobs=request.Jobs,
                Brands=request.Brands,
                mg=request.mg,
                task_machine_mapping={int(k): set(v) for k, v in request.task_machine_mapping.items()}
            )
            logger.info("Environment created successfully")
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error creating environment: {str(e)}")

        # Create QUBO matrix
        logger.info("Creating QUBO matrix...")
        try:
            print(f"env.n: {env.n} env.num_tasks: {env.num_tasks} env.num_machines: {env.num_machines}")
            lambdas = np.array([1]*1000)
            Q = create_qubo_matrix(env, lambdas, env.n + env.num_tasks*(env.n) + len(env.tuples))
            logger.info("QUBO matrix created successfully")
        except Exception as e:
            logger.error(f"Error creating QUBO matrix: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error creating QUBO matrix: {str(e)}")


        bitstring_or = Google_OR(env)

        bitstring = np.array([int(x) for x in bitstring_or])
 
        opt_value = bitstring.T @ Q @ bitstring 
        # Run optimization
        logger.info("Starting optimization...")
        try:
            inst = CompressedVQE(Q, layers=request.layers, na=int(env.n))
            initial_vector = inst.optimize(
                n_measurements=request.n_measurements,
                number_of_experiments=request.number_of_experiments,
                maxiter=request.maxiter
            )
            logger.info("Optimization completed successfully")
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error during optimization: {str(e)}")

        # Get solution
        logger.info("Getting solution...")
        try:
            bitstring = inst.show_solution(shots=1000000)
            logger.info("Solution obtained successfully")
        except Exception as e:
            logger.error(f"Error getting solution: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error getting solution: {str(e)}")

        # Get cost evolution data
        upper_bound, lower_bound, mean = inst.plot_evolution(normalization = [opt_value], label = f"CompressedVQE na={env.n}, shots ={request.n_measurements}")


        cost_evolution = {"upper_bound": [], "lower_bound": [], "mean": []}


        cost_evolution["upper_bound"] = upper_bound
        cost_evolution["lower_bound"] = lower_bound
        cost_evolution["mean"] = mean

        logger.info("Optimization completed successfully")
        return OptimizationResponse(
            solution=bitstring,
            cost_function_value=inst.compute_expectation({bitstring: 1}),
            cost_evolution=cost_evolution
        )

    except Exception as e:
        logger.error(f"Unexpected error during optimization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 