<!-- source: lectures_07_13_pandas_plots_scikit/lecture_07_pandas/reading_material/lec_07c_updated_informed_data_management_modules.ipynb @ 0cc874704aaa -->

# Lecture 07c: Beyond Pandas — Modern Data Frameworks for Python

**Optional for the purpose of this course, necessary if you aim to work with Python and Data Science.**

**The problem:** Pandas is single-threaded and loads everything into RAM. This works for small/medium datasets (up to a few GB), but breaks down when:
- Your dataset is too large to fit in memory.
- Simple operations (groupby, join, filter) take minutes instead of seconds.
- You need to utilize all CPU cores or scale across a cluster.
- You want SQL-style query optimization (projection pushdown, filter pushdown) automatically.

This lecture surveys the modern alternatives, what problem each one solves, and when to reach for which tool.

*Last updated: April 2026.*

## 1. Modern data management modules

**Problem they all share:** Pandas was designed in 2008 for datasets that fit in RAM on a single core.
As datasets grew from megabytes to gigabytes and terabytes, the ecosystem needed tools that:
- Use **all CPU cores** (multi-threading / parallelism).
- Process data **without loading it all into RAM** (out-of-core / streaming / lazy evaluation).
- Apply **query optimization** (only read columns and rows you actually need).
- Support **distributed computing** across machines (for truly massive data).

### 1.2 Multithreading and a new query engine: Polars
[Polars documentation](https://docs.pola.rs/) — [GitHub: 38k+ stars](https://github.com/pola-rs/polars)

**The problem Polars solves:** Pandas is single-threaded and eager — every operation runs immediately, on one core, copying data along the way. Polars was built from scratch in Rust to eliminate these bottlenecks.

**Key features:**
1. **Written in Rust** — compiled, no GIL, no garbage collector overhead. Close-to-metal performance.
2. **Lazy evaluation** — Polars builds a query plan and optimizes it (predicate pushdown, projection pushdown, join reordering) before executing. You get database-level optimization for free.
3. **Multi-threaded by default** — utilizes all CPU cores with zero configuration.
4. **Streaming / out-of-core** — can process datasets larger than RAM by streaming data in chunks.
5. **Apache Arrow memory format** — zero-copy interop with other Arrow-based tools. Polars has its own Arrow-compatible implementation (it does *not* depend on PyArrow).
6. **GPU support (beta)** — optionally run queries on NVIDIA GPUs for maximum in-memory performance.
7. **Strict schema** — data types are known before the query runs, catching errors early.

**When to use Polars over Pandas:**
- Your dataset is too large or too slow for Pandas on a single machine.
- You want automatic query optimization without writing manual pushdowns.
- You need deterministic, strict-schema behavior.

**When Pandas is still fine:**
- Small datasets (a few hundred thousand rows) where Pandas' richer ecosystem and familiarity matter more than speed.

### 1.3 Big data / distributed processing: PySpark
[PySpark documentation](https://spark.apache.org/docs/latest/api/python/)

**The problem PySpark solves:** When your data no longer fits on a single machine — think hundreds of GB to petabytes — you need to distribute it across a cluster. PySpark is the Python API for Apache Spark, the industry standard for distributed data processing.

**Key features:**
1. **Distributed computing** — data is partitioned across a cluster of machines. Computation happens in parallel on each partition.
2. **Lazy evaluation with a query optimizer (Catalyst)** — like Polars, but designed for clusters.
3. **Ecosystem** — includes MLlib (distributed ML), GraphX (graph processing), Spark Streaming (real-time data), and Spark SQL.
4. **Fault tolerance** — if a node fails, Spark recomputes the lost partition from its lineage.
5. **Widely adopted in industry** — most big-data teams use Spark in production.

**When to use PySpark:**
- Your data is too large for a single machine (many GB to TB+).
- You have access to a Hadoop/Spark cluster, Databricks, or a cloud environment.
- You need distributed ML training or real-time stream processing.

**When PySpark is overkill:**
- Datasets under ~10 GB that fit comfortably on a single machine. The overhead of setting up Spark (JVM, cluster coordination) is not worth it for small data. Use Pandas, Polars, or DuckDB instead.

### 1.4 Other notable libraries for large-scale data

**CuPy** — GPU-accelerated NumPy. Offloads numerical computation to NVIDIA GPUs. Not a DataFrame library — it replaces NumPy, not Pandas. Use when you need fast linear algebra or signal processing on the GPU.

**Vaex** — Lazy, out-of-core DataFrames using memory-mapping. Can visualize and explore datasets that exceed RAM without loading them fully. *Note: Vaex development has slowed significantly since 2023; Polars and DuckDB have largely filled its niche.*

**Datatable** — Columnar DataFrame library inspired by R's `data.table`. Multi-threaded, designed for fast aggregation on large datasets. *Note: Originally backed by H2O.ai, development pace has declined. Consider Polars as the modern alternative.*

**Summary:**
| Library | Best for | Status (2026) |
|---------|----------|---------------|
| CuPy | GPU-accelerated NumPy operations | Active, mature |
| Vaex | Memory-mapped exploration of huge files | Low activity |
| Datatable | Fast aggregation à la R's data.table | Low activity |

### 1.5 Other alternatives for distributed processing and parallelism

1. **Dask** — Parallel computing in Python. Extends NumPy, Pandas, and scikit-learn to larger-than-memory datasets. Uses dynamic task scheduling. Good middle ground between "single machine Pandas" and "full cluster Spark." [dask.org](https://dask.org/)

2. **Ray** — General-purpose distributed computing framework. Provides libraries for distributed ML training, hyperparameter tuning, reinforcement learning, and model serving. More flexible than Dask but more complex. [ray.io](https://www.ray.io/)

3. **Modin** — Drop-in replacement for Pandas that uses Ray or Dask under the hood to parallelize operations across all CPU cores. Change one import line (`import modin.pandas as pd`) and get speedups. See Section 1.6 below.

4. **Ibis** — Portable Python DataFrame API that compiles to 20+ backends (DuckDB, Polars, PySpark, BigQuery, Snowflake, PostgreSQL, etc.). Write once, run on any backend by changing one line. See Section 1.8 below. [ibis-project.org](https://ibis-project.org/)

Choose the tool that fits your scale:
- **Single machine, <10 GB**: Pandas, Polars, or DuckDB.
- **Single machine, 10–100 GB**: Polars, DuckDB, or Dask.
- **Cluster, 100 GB – TB+**: PySpark, Dask, or Ray.
- **Portable code across backends**: Ibis.

### 1.6 Scalable pandas code: Modin
[Modin documentation](https://modin.readthedocs.io/en/latest/)

**The problem Modin solves:** You have existing Pandas code and don't want to rewrite it. Modin parallelizes your Pandas operations across all CPU cores with a single import change.

```python
# Before:
import pandas as pd

# After — same API, multi-core execution:
import modin.pandas as pd
```

**Key features:**
1. **Drop-in replacement** — aims for 100% Pandas API compatibility. Your existing code should just work.
2. **Multiple backends** — runs on Ray, Dask, or MPI (via Unidist).
3. **Scales from 1 MB to 1 TB+** — transparent distribution across cores.
4. **Up to 4× speedup on a laptop** with 4 physical cores, just from parallelizing `read_csv`, `groupby`, etc.

**Limitations:**
- Not all Pandas operations are parallelized yet; some fall back to single-threaded Pandas.
- Does not offer query optimization (no lazy evaluation, no pushdowns).
- **In 2023, Snowflake acquired Ponder** (the company behind Modin). Development continues but the project's long-term direction is now tied to Snowflake.

**When to use Modin:** You have a large existing Pandas codebase and want faster execution without rewriting anything.

### 1.7 In-process SQL analytics: DuckDB
[DuckDB documentation](https://duckdb.org/docs/) — [GitHub: 37.6k+ stars](https://github.com/duckdb/duckdb) — [Why DuckDB](https://duckdb.org/why_duckdb)

**The problem DuckDB solves:** You want the power of a SQL database (query optimizer, joins, window functions, aggregations) without the hassle of installing, configuring, and maintaining a database server. DuckDB runs *inside your Python process* — like SQLite, but optimized for analytics instead of transactions.

**Key features:**
1. **Embedded / in-process** — no server to install or maintain. `pip install duckdb` and you're done.
2. **Columnar-vectorized execution** — processes large batches of values at once (not row-by-row like SQLite/Postgres), leading to dramatically faster analytical queries.
3. **Automatic query optimization** — projection pushdown, filter pushdown, join reordering — all happen automatically. You write simple SQL; DuckDB figures out the fastest execution plan.
4. **Zero-copy on Pandas/Arrow** — DuckDB can query a Pandas DataFrame *directly* without copying or importing data. "Pandas-in, Pandas-out."
5. **Multi-threaded** — uses all available CPU cores by default.
6. **Reads anything** — CSV, Parquet, JSON, Arrow, Excel, and even remote files over HTTP/S3. You can query a Parquet file on S3 without downloading it first.
7. **Persistent or in-memory** — store data in a single `.duckdb` file, or use it purely in-memory for ad-hoc analysis.
8. **Full SQL support** — window functions, CTEs (including recursive), correlated subqueries, PIVOT/UNPIVOT, nested types, and more (uses the PostgreSQL SQL parser).
9. **Extensible** — Parquet, JSON, HTTP/S3, and spatial support are all implemented as extensions.

**DuckDB vs. Pandas — a concrete example:**
```python
import duckdb
import pandas as pd

# DuckDB queries the Pandas DataFrame directly — no import step!
df = pd.read_csv("big_file.csv")
result = duckdb.query("SELECT category, AVG(price) FROM df GROUP BY category").to_df()
```
In benchmarks, DuckDB is **2–30× faster than Pandas** on grouped aggregates, filtered aggregates, and joins — because its optimizer combines filter + project + aggregate into a single pass, while Pandas creates intermediate copies at every step.

**DuckDB vs. SQLite:**
- SQLite processes rows one at a time (OLTP-optimized). DuckDB processes columns in vectors (OLAP-optimized).
- For analytical queries (aggregations, scans, joins over large tables), DuckDB is orders of magnitude faster.

**When to use DuckDB:**
- You love SQL and want to query CSV/Parquet/DataFrames with it.
- Your analytical queries are slow in Pandas and you want automatic optimization.
- You want a single-file database for analytics (like SQLite for OLAP).
- You need to query remote Parquet files on S3 or HTTP without downloading them.

**When DuckDB is not the right tool:**
- High-concurrency transactional workloads (many simultaneous writes) — use PostgreSQL.
- Distributed processing across a cluster — use PySpark or Dask.

### 1.8 Portable DataFrame API: Ibis
[Ibis documentation](https://ibis-project.org/) — [GitHub](https://github.com/ibis-project/ibis)

**The problem Ibis solves:** Every data tool has its own API — Pandas, Polars, PySpark, SQL dialects. If you write code for one, you have to rewrite it for another. Ibis provides a *single Python DataFrame API* that compiles to 20+ backends.

**Key idea:**
```python
import ibis

# Develop locally with DuckDB (fast, no setup):
con = ibis.connect("duckdb://")
t = con.read_parquet("data.parquet")
result = t.group_by("category").agg(avg_price=t.price.mean())

# Deploy to BigQuery by changing ONE line:
con = ibis.connect("bigquery://my-project/my-dataset")
```

**Supported backends include:** DuckDB (default), Polars, DataFusion, PySpark, PostgreSQL, MySQL, SQLite, BigQuery, Snowflake, Databricks, ClickHouse, Trino, and more.

**When to use Ibis:**
- You want to prototype locally (DuckDB/Polars) and deploy to a warehouse (BigQuery/Snowflake) without rewriting.
- You work across multiple data systems and want one consistent API.
- You want to compose Python DataFrame expressions and SQL freely (Ibis can show you the generated SQL).

## 2. Head-to-head comparisons

Performance benchmarks: [Polars benchmarks](https://pola.rs/posts/benchmarks/) — [DuckDB db-benchmark](https://duckdblabs.github.io/db-benchmark/)

### 2.1 Polars vs. Modin

| Dimension | Polars | Modin |
|-----------|--------|-------|
| **Approach** | New engine from scratch (Rust) | Parallelizes existing Pandas |
| **API** | Own API (similar but not identical to Pandas) | Drop-in Pandas replacement |
| **Query optimization** | Yes — lazy evaluation, pushdowns | No |
| **Memory efficiency** | Streaming/out-of-core, Arrow-based | Loads entire dataset in memory |
| **Multi-threading** | Built-in from the ground up | Via Ray/Dask backend |
| **Learning curve** | Must learn Polars expressions | Zero — uses Pandas API |
| **Backing** | Independent open source | Snowflake (via Ponder acquisition) |

**Bottom line:** Polars is faster and more memory-efficient. Modin is easier to adopt if you have existing Pandas code.

### 2.2 Polars vs. DuckDB

Both are fast, single-machine, Arrow-based analytical engines. The key difference is the **interface**:

| Dimension | Polars | DuckDB |
|-----------|--------|--------|
| **Primary interface** | Python DataFrame API | SQL |
| **Execution** | Lazy DataFrame expressions | SQL queries with a columnar-vectorized engine |
| **Optimization** | Query plan optimizer on expressions | Full SQL query optimizer (Catalyst-style) |
| **Zero-copy interop** | Arrow-native | Reads Pandas/Arrow DataFrames directly |
| **Persistence** | In-memory (write to Parquet/CSV for storage) | Single-file `.duckdb` database |
| **Best for** | Python-centric data pipelines | SQL-centric analytics, ad-hoc queries on files |

**Zero-copy interop between Polars ↔ DuckDB:** they can exchange data without copying. [DuckDB-Polars integration guide](https://duckdb.org/docs/guides/python/polars.html)

**Bottom line:** If you think in DataFrames → Polars. If you think in SQL → DuckDB. Many practitioners use both.

### 2.3 Quick decision guide

| Your situation | Recommended tool |
|----------------|-----------------|
| Small dataset, learning Python | **Pandas** |
| Medium dataset, need speed, Python-centric | **Polars** |
| Any size, you prefer SQL | **DuckDB** |
| Existing Pandas code, need quick speedup | **Modin** |
| Huge data, need a cluster | **PySpark** |
| Write once, run on any backend | **Ibis** |
| GPU-accelerated numerical computing | **CuPy** |

*There is no "best" tool. Pick the one that matches your data size, skill set, and deployment environment.*

## 3. The Apache Arrow ecosystem and PyArrow in Pandas
[Apache Arrow](https://arrow.apache.org/) — [PyArrow docs](https://arrow.apache.org/docs/python/index.html) — [Pandas PyArrow docs](https://pandas.pydata.org/docs/user_guide/pyarrow.html)

**What is Apache Arrow?** A language-independent, columnar memory format designed for efficient in-memory analytics. It serves as the **universal data interchange layer** — allowing tools like Pandas, Polars, DuckDB, Spark, and R to share data without copying.

**Since Pandas 2.0 (April 2023):** Pandas supports Arrow-backed columns via `dtype_backend="pyarrow"`:
```python
df = pd.read_csv("data.csv", dtype_backend="pyarrow")  # Arrow dtypes!
```
This gives Pandas native support for missing values (no more sentinel NaN for integers!), strings as Arrow strings (much faster than Python objects), and zero-copy exchange with other Arrow tools.

**Why this matters:** Arrow is the common ground that makes Polars ↔ DuckDB ↔ Pandas ↔ Spark interop efficient. It is one of the most important infrastructure projects in the data ecosystem.

## 4. Staying up-to-date on the Python data ecosystem

### 4.1. Read the official docs.   
e.g. https://pandas.pydata.org/

### 4.2 Conferences: PyData, PyCon, SciPy

Follow the [PyData community](https://pydata.org/) — conferences with practical talks on data tools.

- [PyData Global](https://pydata.org/global/) — annual online conference.
- [PyCon](https://pycon.org/) — the main Python conference (US, EU, and regional editions).
- [SciPy](https://www.scipy.org/conferences.html) — Scientific Python conference.

Past talks on YouTube: [PyDataTV](https://www.youtube.com/@PyDataTV)

### 4.3 Ask questions in stackoverflow.   
[How to ask a good question](https://stackoverflow.com/help/how-to-ask)

### 4.4 Follow the right people and communities
- **Wes McKinney** — creator of Pandas, now leading Apache Arrow/Voltron Data.
- **Ritchie Vink** — creator of Polars.
- **Hannes Mühleisen & Mark Raasveldt** — creators of DuckDB.
- **Cassie Kozyrkov** — decision intelligence, data science communication.
- Communities: [r/datascience](https://www.reddit.com/r/datascience/), [PyData Slack](https://pydata.org/), DuckDB Discord, Polars Discord.

### 4.5 Double-check AI copilots and assistants
AI coding assistants are useful but **not infallible**. They can confidently state things that are outdated or wrong.   
Always verify against official documentation and changelogs. The example below is real.

`argythana:`  
> append_df = df1.append(df2)  AttributeError: 'DataFrame' object has no attribute 'append'
> Why is this wrong?   


`GitHub Copilot:`
> The error message you're seeing typically indicates that the method you're trying to use doesn't exist for the object you're using it on. However, the `append` method is a valid method for a DataFrame object in pandas.
> The error might be due to a different reason such as:
> 1. You might have overwritten the pandas DataFrame class with a variable of the same name. Make sure you haven't used `DataFrame` as a variable elsewhere in your code.
> 2. There might be an issue with your pandas installation. Try reinstalling pandas using pip or conda.
> 3. You might be using an older version of pandas where the `append` method is not available. You can check your pandas version with `print(pd.__version__)`. If it's not the latest version, consider upgrading it.
> If none of these are the case, please provide more context or check the rest of your code for potential issues.

`argythana:`   

> when was df.append() method deprecated and when whas it totally removed from pandas?

`GitHub Copilot:`
> The `append()` method is not deprecated and has not been removed from pandas. It is still a valid method for appending rows of one DataFrame to the end of another DataFrame, returning a new object. The error you're encountering might be due to a different reason. Please ensure that `df1` and `df2` are indeed pandas DataFrames. If the issue persists, consider checking your pandas installation or version.

`argythana (April 2024):`  
> Which version of GitHub Copilot is this?

`GitHub Copilot:`  
> This is the GPT-4 version of GitHub Copilot.

*Even GPT-4 got this wrong. Always cross-check with official docs!*

**Meanwhile, since pandas 1.4 (January 22, 2022) `df.append()` has been [deprecated](https://pandas.pydata.org/pandas-docs/version/1.4/reference/api/pandas.DataFrame.append.html).**        


**`df.append()` has been removed for ages, since April 3, 2023.**   

Read this [reply.](https://stackoverflow.com/a/75956237)
