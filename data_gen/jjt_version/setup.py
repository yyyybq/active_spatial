from setuptools import setup, find_packages

# Define extras here so we can derive "all" programmatically
extras = {
    # Evaluation pipeline (single-backend runner + adapters)
    "eval": [
        # Orchestration / config
        "hydra-core>=1.3.2,<1.4",
        "omegaconf>=2.3,<2.4",

        # OpenAI-compatible clients (OpenAI/Azure/Together/vLLM/sglang all reuse this SDK)
        "openai>=1.40.0,<2",

        # Closed-source providers
        "anthropic>=0.34.0,<1",          # Claude (AsyncAnthropic)
        "google-generativeai>=0.7.2,<1", # Gemini

        # Vision utilities for PIL <-> PNG data URLs
        "Pillow>=10.0.0,<12",
    ],

    # You can add more extras later, e.g., "dev": [...], "render": [...]
}

# "all" aggregates every extra listed above (unique + sorted)
extras["all"] = sorted({pkg for group in extras.values() for pkg in group})

setup(
    name="view_suite",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        # Core runtime deps for your package (keep your originals)
        "numpy",
        "requests",
        "open3d==0.19.0",
        "uvicorn",
        "fastapi",
        "websockets==15.0.1",
        "gsplat",
        "fire",
        "plyfile",
        

    ],
    extras_require=extras,
)
