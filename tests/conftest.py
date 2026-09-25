import os

os.environ["RAGBENCH_STORE_BACKEND"] = "memory"
os.environ["RAGBENCH_EXPERIMENT_DB_PATH"] = ":memory:"
os.environ["OPENAI_API_KEY"] = ""
