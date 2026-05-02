import cProfile
import pstats
from src.model import PLaplacianModel
from src.solver import PLaplacianSolver

def run_profiled():
    model = PLaplacianModel(p=6, h=1.0, Nx=1000, epsilon=1e-6)
    solver = PLaplacianSolver(model)
    results, stats = solver.solve([0.05], method="CVODE", sparse=True,
                                  rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_profiled()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)
