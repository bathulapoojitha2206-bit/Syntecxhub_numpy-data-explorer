# ⚡ NumPy Data Explorer — Interactive Numerical Computing Lab

A complete, production-grade, interactive numerical computing laboratory and exploratory dashboard built with **Python 3**, **NumPy**, **Pandas**, **Plotly**, and **Streamlit**.

---

## 📖 1. Project Overview & Problem Statement

### The Challenge
When developers and data analysts transition to numerical computing and data science, they frequently encounter mental blocks:
- Treating NumPy arrays like standard Python lists, leading to slow $O(N)$ Python-level `for` loops.
- Writing code plagued by subtle, silent bugs caused by mutating array **views** instead of independent **copies**.
- Struggling with multidimensional **axis mechanics** (`axis=0` vs `axis=1` vs `axis=None`).
- Confusion over **broadcasting rules** when combining arrays with differing ranks and shapes.
- Inefficient RAM utilization caused by boxed Python `PyObject` pointer indirection.

### The Solution: NumPy Data Explorer
**NumPy Data Explorer** transforms abstract array theory into an interactive, visual, and empirical laboratory. Users can manipulate matrices in real time, inspect memory buffers and strides, track view mutation, verify broadcasting compatibility, execute verified micro-benchmarks, and perform end-to-end exploratory data analysis on real-world datasets.

---

## 🎯 2. Learning Objectives
1. **Contiguous Memory Buffers**: Understand how `ndarray` stores homogeneous binary buffers in C-order or Fortran-order.
2. **View vs Copy Semantics**: Visually and empirically observe how basic slices share memory and how `.copy()` isolates data.
3. **Axis Collapsing**: Demystify `axis=0` (down the rows, collapsing to column aggregates) and `axis=1` (across columns, collapsing to row aggregates).
4. **The 4 Rules of Broadcasting**: Trace trailing-to-leading right-to-left shape alignment with step-by-step diagnostics.
5. **Secure Numerical File I/O**: Master `.npy`, `.npz` (multi-array/compressed), and `.csv` formats while enforcing `allow_pickle=False`.
6. **SIMD & Hardware Vectorization**: Measure real CPU execution speedups and RAM memory reduction compared to Python lists.

---

## ✨ 3. Feature Highlights by Lab Module

### 🧪 1. Array Creation Lab
- Explore constructors: `np.array()`, `np.zeros()`, `np.ones()`, `np.empty()`, `np.arange()`, `np.linspace()`, `np.eye()`, and `np.full()`.
- Modern PRNG generation with `np.random.default_rng(seed)` for uniform, Gaussian normal, and integer distributions.
- Explicit safety warning on `np.empty()` detailing uninitialized raw RAM garbage bits.
- Real-time array metadata inspector: shape, ndim, size, itemsize, nbytes, strides, contiguity flags, and memory status.
- Instant copy-pasteable Python/NumPy code generator.

### 🔍 2. Indexing & Slicing Lab
- Positive, negative, and step slicing across 1D vectors and 2D matrices.
- Row and column extractions with interactive slice boundaries.
- Boolean masking and conditional filtering (`>`, `<`, `between`, `even`, `odd`).
- Fancy indexing with arbitrary integer lists, diagonal extraction (`np.diag`), and array reversal (`[::-1]`).
- **Interactive View vs Copy Mutation Experiment**: Mutate sliced segments live to observe in-place original array alteration versus `.copy()` protection.

### ➕ 3. Mathematical Operations
- Element-wise binary arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`.
- IEEE 754 safe arithmetic: handles division by zero ($1/0 \rightarrow +\infty$, $0/0 \rightarrow \text{NaN}$) with clear explanations.
- Unary functions: `sqrt`, `abs`, `exp`, `log`, `log10`, `log2`, `round`, `floor`, `ceil`, `sin`, `cos`.
- Negative square roots and non-positive logarithms gracefully flagged with NaN warnings.
- Vectorized conditional logic via `np.where(condition, x, y)` and element-wise min/max (`np.maximum`, `np.minimum`).

### 📊 4. Statistical & Axis Lab
- Exhaustive metrics: `sum`, `mean`, `median`, `std`, `var`, `min`, `max`, `ptp` (peak-to-peak), `percentile`, `quantile`, `argmin`, `argmax`, `cumsum`, and `cumprod`.
- Complete axis mechanics: `axis=None` (global flattened), `axis=0` (down rows -> column aggregates), `axis=1` (across columns -> row aggregates).
- Degrees of freedom (`ddof`) explained: population variance ($\text{ddof}=0$, NumPy default) versus sample variance ($\text{ddof}=1$, Bessel's correction / Pandas default).
- Interactive distribution histograms with mean/median vertical guide lines.

### 🔄 5. Reshaping & Manipulation
- Array reshaping with automated `-1` dimension inference and compatibility validation (`prod(new_shape) == size`).
- In-depth comparison of `ravel()` (zero-copy view) vs `flatten()` (guaranteed deep copy).
- Transposition (`.T` / `np.transpose`).
- Joining and stacking: `concatenate`, `vstack`, `hstack`, `stack`.
- Splitting: `np.split` (equal cuts) versus `np.array_split` (unequal segment tolerance).
- Dimension expansions (`expand_dims`, `squeeze`), block tiling (`tile`), and element repetitions (`repeat`).

### 📡 6. Broadcasting Visualizer
- Automated 4-Rule Broadcasting Engine:
  1. Compares dimensions from right to left (trailing to leading).
  2. Left-pads smaller shapes with 1s.
  3. Checks if dimensions are equal or either equals 1.
  4. Computes resulting broadcast shape or raises informative conflict diagnostics.
- Real-time diagnostic table showing per-axis stretch steps.
- Real-world scenarios: Scalar-to-Matrix, Row-Vector-to-Matrix, Column-to-Row Outer Product, and Feature Mean Normalization.

### 💾 7. NumPy File Operations
- **Binary `.npy`**: Single-array serialization with zero precision loss.
- **Archive `.npz`**: Multi-array bundling with optional `savez_compressed` DEFLATE compression.
- **Plaintext `.csv` / `.txt`**: Human-readable delimited exports.
- **Direct Browser Downloads**: In-memory byte streaming without exposing server filesystem paths.
- **Secure File Deserialization**: Strictly enforces `allow_pickle=False`, verifies binary magic signatures (`\x93NUMPY` and `PK\x03\x04`), and enforces file size boundaries.

### ⚡ 8. Performance Benchmark
- Micro-benchmark suite comparing Python lists against NumPy arrays for creation, addition, multiplication, summation, and scalar scaling.
- Fair measurement methodology: warm-up iterations, multiple trials, and median execution time reporting.
- Numerical equivalence verification to guarantee identical calculations.
- RAM memory footprint estimation (PyObject pointer array vs contiguous unboxed buffer).
- Interactive Plotly speedup bar charts and logarithmic scaling curves ($N = 1,000$ to $N = 500,000$).

### 📈 9. Practical Data Analysis
- Realistic synthetic datasets:
  - **Student Marks**: 15 students across 5 subjects.
  - **Monthly Regional Sales**: 12 months across 4 territories.
  - **Product Quality Ratings**: 10 products across 4 audit criteria.
- Analytical workflows: Min-Max feature normalization, high-performer conditional masking, entity rankings, and category breakdowns.

---

## 🛠️ 4. Technology Stack
- **Language**: Python 3.11 / 3.12
- **Numerical Core**: NumPy ($\ge 1.24.0$)
- **DataFrames & I/O**: Pandas ($\ge 2.0.0$)
- **Web Dashboard**: Streamlit ($\ge 1.28.0$)
- **Interactive Charts**: Plotly ($\ge 5.17.0$) & Matplotlib ($\ge 3.7.0$)
- **Automated Testing**: Pytest ($\ge 7.4.0$)

---

## 📁 5. Project Directory Structure

```
numpy-data-explorer/
├── app.py                     # Main interactive Streamlit application
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation and guide
├── .gitignore                 # Standard Python/IDE/cache exclusions
├── src/                       # Modular numerical computing logic
│   ├── __init__.py            # Package initialization
│   ├── array_creation.py      # Array constructors & metadata extraction
│   ├── indexing_slicing.py    # Slicing, boolean masks & view mutators
│   ├── mathematical_operations.py # Arithmetic, unary math & np.where
│   ├── statistical_operations.py  # Statistical metrics, axes & ddof
│   ├── reshaping.py           # Reshaping, ravel vs flatten, stacking
│   ├── broadcasting.py        # 4-rule broadcasting diagnostic engine
│   ├── file_operations.py     # Secure .npy, .npz, .csv serialization
│   ├── performance_benchmark.py # Micro-benchmarks & memory footprint
│   └── visualizations.py      # Plotly heatmaps, distributions & charts
├── data/                      # Synthetic analytical datasets
│   ├── student_marks.csv
│   ├── monthly_sales.csv
│   ├── product_ratings.csv
│   └── sample_matrix.npy
├── outputs/                   # Export directory
└── tests/                     # Automated Pytest suite (51 test cases)
    ├── __init__.py
    ├── test_array_creation.py
    ├── test_array_operations.py
    ├── test_reshaping_broadcasting.py
    ├── test_file_operations.py
    └── test_performance.py
```

---

## 🚀 6. Installation & Local Setup

### Step 1: Clone or Navigate to Repository
```bash
git clone https://github.com/bathulapoojitha/numpy-data-explorer.git
cd numpy-data-explorer
```

### Step 2: Create and Activate Virtual Environment (Recommended)
```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## ⚡ 7. Empirical Performance Benchmark Results

Measured directly on the host system with $N = 100,000$ elements using median execution timings and warm-up passes:

| Operation | Input Size ($N$) | Python List Time | NumPy Array Time | Measured Speedup | Verified Accurate? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Element-wise Multiplication** | 100,000 | 3.87 ms | 0.049 ms | **🚀 79.2x faster** | ✅ Yes |
| **Scalar Multiplication** | 100,000 | 2.35 ms | 0.040 ms | **🚀 58.5x faster** | ✅ Yes |
| **Element-wise Addition** | 100,000 | 3.47 ms | 0.069 ms | **🚀 50.6x faster** | ✅ Yes |
| **Summation (`sum`)** | 100,000 | 0.285 ms | 0.038 ms | **🚀 7.4x faster** | ✅ Yes |

### 🧠 RAM Memory Footprint Comparison ($N = 100,000$ Floats):
- **Python List (`list[float]`)**: `3.05 MB` (~32 bytes per element: pointer + PyObject header)
- **NumPy Array (`float64`)**: `781.36 KB` (exactly 8 bytes per contiguous IEEE 754 float)
- **Memory Efficiency**: NumPy uses **4.0x less RAM**!

---

## 🧪 8. Automated Testing

The project includes an exhaustive Pytest suite covering all modules, boundary conditions, zero-divisions, view mutations, and serialization round-trips.

To run the test suite:
```bash
pytest -v
```

### Test Suite Execution Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.5, pytest-9.1.1
collected 51 items

tests/test_array_creation.py ...........                                 [ 21%]
tests/test_array_operations.py ...............                           [ 50%]
tests/test_file_operations.py ......                                     [ 62%]
tests/test_performance.py ........                                       [ 78%]
tests/test_reshaping_broadcasting.py ...........                         [100%]

============================= 51 passed in 0.84s ==============================
```

---

## ☁️ 9. Deployment to Render

This application is ready for deployment on **Render** (or any containerized cloud PaaS).

### Render Web Service Configuration:
- **Environment**: Python 3
- **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
  ```

### Steps to Deploy:
1. Push your repository to GitHub.
2. Sign in to [Render Dashboard](https://dashboard.render.com).
3. Click **New +** -> **Web Service** -> Connect your GitHub repository (`numpy-data-explorer`).
4. Paste the Build Command and Start Command specified above.
5. Choose the **Free** instance tier and click **Deploy Web Service**.
6. Render will automatically bind the `$PORT` environment variable and issue an HTTPS public URL.

---

## 🔒 10. Security & Best Practices
- **No Arbitrary Code Execution**: The application uses structured UI inputs rather than evaluating untrusted user-submitted Python strings.
- **Safe File Deserialization**: All NumPy file loading strictly enforces `allow_pickle=False` to prevent arbitrary code execution vulnerabilities.
- **Resource Protection**: Maximum array bounds and benchmark element counts are capped to prevent server out-of-memory (OOM) crashes.

---

## 🔮 11. Limitations & Future Roadmap
- **Current Limitations**: 3D tensor visualizations are rendered via 2D depth slices rather than volumetric voxel renders.
- **Future Improvements**:
  - Add GPU acceleration benchmarking via CuPy.
  - Implement linear algebra lab (eigenvalues, SVD, Cholesky decompositions).
  - Add interactive Einstein summation (`np.einsum`) visualizer.

---

## 📄 License
This project is open-source under the MIT License.
