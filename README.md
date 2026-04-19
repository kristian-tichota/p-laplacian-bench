# Overview

The purpose of this repository is transparency and reproducible research. Parts of the code are used in my bachelor's thesis for numerical evaluations to compare against analytic bounds.

<img width="600" alt="A parameter sweep image representing the repository" src="https://github.com/user-attachments/assets/fc804a6f-1f6c-4a91-b4cf-cd1140c76d61" />

# Installation
First, clone the repository:
```bash
git clone https://github.com/kristian-tichota/p-laplacian-bench
cd p-laplacian-bench
```
It is recommended to isolate project dependencies within a dedicated Python virtual environment (e.g. `venv` or `conda`). If using the built-in `venv` module:
```bash
python -m venv .venv
source .venv/bin/activate
```
Once the environment is activated, install the necessary packages:
```bash
pip install -r requirements.txt
```

# Benchmarking
The file `benchmark.org` serves as a literate programming dashboard for comfortably executing benchmarks. It briefly explains the parameters, metrics, and usage via examples. The benchmarks provide sample empirical comparisons (e.g. sparse vs non-sparse implementations) that validate theoretical claims and show via examples how to define the parameter space for custom benchmarks. There is also a brief tutorial section dedicated to replicating the results from the LaTeX article.

**For Emacs Users:**
1. Open `benchmark.org` in Emacs.
2. Ensure Org-Babel is configured for Bash.
3. Run `M-x org-babel-execute-buffer` to reproduce all results.

Non-Emacs users can convert `benchmark.org` into a `.ipynb` file for use in Jupyter or run the Python commands directly by copying them from `benchmark.org`.

# Result Contingency
Benchmark comparisons are highly dependent on the host architecture. Variations in L1/L2/L3 cache topologies, memory latency, CPU instruction sets, OS kernel, etc., can fundamentally alter relative execution speeds. Executing the benchmarks on different hardware may not simply scale the execution time. A detailed description of the testing host architecture can be found in the LaTeX document. 

# Licence
* Source code, scripts, and build files: AGPL-3.0-or-later.
* Documentation, prose, illustrations, and other non-code copyrightable assets: CC-BY-SA-4.0.

# LLM Use Disclaimer
Large language models have assisted with code writing to alleviate typing strain due to severe finger tendinosis. The underlying logic, architecture, and validation of all code remain my original work. I take full responsibility for the codebase's correctness.
