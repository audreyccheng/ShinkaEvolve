"""
Evaluator for EPLB (Expert Parallelism Load Balancer) example
"""

import os
import argparse
import sys
import time
import json
import importlib.util
import torch
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval

# Global list to store execution times during evaluation
_execution_times: List[float] = []

# Add the openevolve_examples directory to the path to import evaluator utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'openevolve_examples', 'eplb'))

# Import evaluation utilities from openevolve example
WORKLOAD_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'openevolve_examples', 'eplb', 'expert-load.json')

# Check if workload file exists before importing
if not os.path.exists(WORKLOAD_PATH):
    print(f"Warning: Workload file not found at {WORKLOAD_PATH}")
    print("Please download expert-load.json from https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json")
    print("and place it in the openevolve_examples/eplb directory")
    print("Running with dummy workloads for testing purposes.")
    WORKLOAD_PATH = None
    load_workloads = None
    simulate_inference = None
else:
    try:
        from evaluator import load_workloads, simulate_inference
    except ImportError:
        print("Warning: Could not import evaluator utilities from openevolve_examples/eplb")
        WORKLOAD_PATH = None
        load_workloads = None
        simulate_inference = None

# EPLB configuration
NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4


def dummy_simulate_inference(log2phy: torch.Tensor, logcnt: torch.Tensor, workload: torch.Tensor) -> float:
    """Dummy simulation when real evaluator is not available."""
    # Simple heuristic: compute load variance as a proxy for balancedness
    num_layers, num_logical_experts = workload.shape
    num_physical_experts = NUM_REPLICAS
    
    total_physical_load = torch.zeros(num_layers, num_physical_experts, dtype=torch.float)
    
    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            logical_load = workload[layer_id][logical_id].item()
            if logical_load <= 0:
                continue
            
            num_replicas = int(logcnt[layer_id][logical_id].item())
            if num_replicas <= 0:
                continue
            
            physical_ids = log2phy[layer_id][logical_id][:num_replicas]
            replica_load = logical_load / num_replicas
            total_physical_load[layer_id, physical_ids] += replica_load
    
    total_load = total_physical_load.sum()
    if total_load == 0:
        return 0.0
    
    layer_avg = total_physical_load.mean(dim=1)
    layer_max = total_physical_load.max(dim=1).values
    
    avg_load = layer_avg.sum().item()
    max_load = layer_max.sum().item()
    
    balancedness = avg_load / max_load if max_load > 0 else 0.0
    return balancedness


def format_results_string(balancedness: float, speed: float) -> str:
    """Formats EPLB results into a multi-line string for display."""
    return f"  balancedness_score = {balancedness:.6f}\n  speed_score = {speed:.6f}"


def adapted_validate_eplb(
    run_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    atol=1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Validates EPLB results based on the output of 'run_eplb'.

    Args:
        run_output: Tuple (phy2log, log2phy, logcnt) from run_eplb.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    phy2log, log2phy, logcnt = run_output
    msg = "The EPLB mapping is valid and properly structured."
    
    # Check that outputs are tensors
    if not isinstance(phy2log, torch.Tensor):
        msg = f"phy2log should be a torch.Tensor, got {type(phy2log)}"
        return False, msg
    
    if not isinstance(log2phy, torch.Tensor):
        msg = f"log2phy should be a torch.Tensor, got {type(log2phy)}"
        return False, msg
    
    if not isinstance(logcnt, torch.Tensor):
        msg = f"logcnt should be a torch.Tensor, got {type(logcnt)}"
        return False, msg
    
    # Check tensor shapes are reasonable
    if len(phy2log.shape) != 2:
        msg = f"phy2log should be 2D tensor, got shape {phy2log.shape}"
        return False, msg
    
    if len(log2phy.shape) != 3:
        msg = f"log2phy should be 3D tensor, got shape {log2phy.shape}"
        return False, msg
    
    if len(logcnt.shape) != 2:
        msg = f"logcnt should be 2D tensor, got shape {logcnt.shape}"
        return False, msg
    
    # Check for NaN values
    if torch.isnan(phy2log).any():
        msg = "NaN values detected in phy2log"
        return False, msg
    
    if torch.isnan(log2phy).any():
        msg = "NaN values detected in log2phy"
        return False, msg
    
    if torch.isnan(logcnt).any():
        msg = "NaN values detected in logcnt"
        return False, msg
    
    # Check for negative values in count
    if (logcnt < 0).any():
        msg = "Negative values found in logcnt"
        return False, msg
    
    return True, msg


def get_eplb_kwargs(run_index: int) -> Dict[str, Any]:
    """
    Provides keyword arguments for EPLB runs.
    
    Each run uses a different workload from the workload history.
    """
    if WORKLOAD_PATH and os.path.exists(WORKLOAD_PATH):
        workloads = load_workloads(WORKLOAD_PATH)
        if run_index < len(workloads) - 1:
            return {
                "weight": workloads[run_index],
                "num_replicas": NUM_REPLICAS,
                "num_groups": NUM_GROUPS,
                "num_nodes": NUM_NODES,
                "num_gpus": NUM_GPUS,
            }
    
    # Return dummy workload if file not available
    return {
        "weight": torch.randn(64, 256),  # Dummy workload
        "num_replicas": NUM_REPLICAS,
        "num_groups": NUM_GROUPS,
        "num_nodes": NUM_NODES,
        "num_gpus": NUM_GPUS,
    }


def timed_experiment_wrapper(experiment_fn):
    """
    Wraps an experiment function to measure and record execution time.
    Times are stored in the global _execution_times list.
    """
    def wrapper(*args, **kwargs):
        global _execution_times
        start_time = time.perf_counter()
        result = experiment_fn(*args, **kwargs)
        end_time = time.perf_counter()
        _execution_times.append(end_time - start_time)
        return result
    return wrapper


def aggregate_eplb_metrics(
    results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
    results_dir: str
) -> Dict[str, Any]:
    """
    Aggregates metrics for EPLB across multiple workload runs.
    
    Uses actual execution times from _execution_times global to compute speed_score.
    Speed score formula matches OpenEvolve: speed_score = 0.02 / avg_time
    """
    global _execution_times
    
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    if WORKLOAD_PATH and os.path.exists(WORKLOAD_PATH) and load_workloads is not None:
        workloads = load_workloads(WORKLOAD_PATH)
        sim_func = simulate_inference if simulate_inference is not None else dummy_simulate_inference
        
        balancedness_scores = []
        for i, (phy2log, log2phy, logcnt) in enumerate(results):
            if i + 1 < len(workloads):
                # Evaluate on the next workload (predictive balancing)
                balancedness = sim_func(log2phy, logcnt, workloads[i + 1])
                balancedness_scores.append(balancedness)
        
        avg_balancedness = sum(balancedness_scores) / len(balancedness_scores) if balancedness_scores else 0.0
        
        # Calculate speed_score from actual execution times
        # Formula matches OpenEvolve: speed_score = 0.02 / avg_time
        if _execution_times:
            avg_time = sum(_execution_times) / len(_execution_times)
            speed_score = 0.02 / avg_time if avg_time > 0 else 0.0
            print(f'avg_time: {avg_time:.6f}s, speed_score: {speed_score:.6f}')
        else:
            # Fallback if no times recorded (shouldn't happen normally)
            print("Warning: No execution times recorded, using speed_score = 0.0")
            speed_score = 0.0
            avg_time = 0.0
        
        combined_score = (avg_balancedness + speed_score) / 2
        
        public_metrics = {
            "results_str": format_results_string(avg_balancedness, speed_score),
            "num_workloads_evaluated": len(balancedness_scores),
            "avg_execution_time": float(avg_time) if _execution_times else None,
        }
        private_metrics = {
            "avg_balancedness_score": float(avg_balancedness),
            "speed_score": float(speed_score),
        }
    else:
        # Dummy metrics if workload file not available
        avg_balancedness = 0.5
        speed_score = 0.0  # Changed from 1.0 - no valid speed without real workload
        combined_score = avg_balancedness / 2  # Only count balancedness
        
        public_metrics = {
            "results_str": "Workload file not available for evaluation",
            "num_workloads_evaluated": 0,
        }
        private_metrics = {
            "avg_balancedness_score": float(avg_balancedness),
            "speed_score": float(speed_score),
        }
    
    metrics = {
        "combined_score": float(combined_score),
        "public": public_metrics,
        "private": private_metrics,
    }

    return metrics


def main(program_path: str, results_dir: str):
    """Runs the EPLB evaluation with proper timing measurement."""
    global _execution_times
    
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    # Clear execution times from any previous runs
    _execution_times = []

    # Number of workload samples to evaluate
    num_experiment_runs = 5 if WORKLOAD_PATH and os.path.exists(WORKLOAD_PATH) else 1

    # Load the program and get the experiment function
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load program from {program_path}")
        return
    
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    if not hasattr(program, "run_eplb"):
        # Try rebalance_experts as fallback
        if hasattr(program, "rebalance_experts"):
            # Create a wrapper that matches expected interface
            def run_eplb_wrapper(**kwargs):
                return program.rebalance_experts(
                    kwargs["weight"],
                    kwargs["num_replicas"],
                    kwargs["num_groups"],
                    kwargs["num_nodes"],
                    kwargs["num_gpus"],
                )
            experiment_fn = run_eplb_wrapper
        else:
            print("Error: Program does not have 'run_eplb' or 'rebalance_experts' function")
            return
    else:
        experiment_fn = program.run_eplb
    
    # Run experiments with timing
    results = []
    all_valid = True
    error_msg = None
    
    for run_idx in range(num_experiment_runs):
        kwargs = get_eplb_kwargs(run_idx)
        
        # Measure execution time
        start_time = time.perf_counter()
        try:
            result = experiment_fn(**kwargs)
            end_time = time.perf_counter()
            _execution_times.append(end_time - start_time)
            
            # Validate result
            is_valid, validation_msg = adapted_validate_eplb(result)
            if not is_valid:
                all_valid = False
                error_msg = validation_msg
                print(f"Run {run_idx}: Validation failed - {validation_msg}")
            else:
                results.append(result)
                print(f"Run {run_idx}: Success (time: {end_time - start_time:.4f}s)")
                
        except Exception as e:
            end_time = time.perf_counter()
            all_valid = False
            error_msg = str(e)
            print(f"Run {run_idx}: Error - {e}")
    
    # Aggregate metrics
    metrics = aggregate_eplb_metrics(results, results_dir)
    
    # Save results
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(results_dir, "correct.json"), "w") as f:
        json.dump({"correct": all_valid, "error": error_msg}, f, indent=2)

    if all_valid:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EPLB evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_eplb')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)

