from setuptools import setup, find_packages

setup(
    name="neuromm2026",
    version="1.0.0",
    description="NeuroMM-2026 — EEGMamba Y-Architecture for Multimodal Seizure Detection",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "mamba": ["mamba-ssm>=2.0.4", "causal-conv1d>=1.2.0"],
        "lora":  ["peft>=0.8.0"],
        "wandb": ["wandb>=0.16.0"],
        "dev":   ["pytest>=7.4.0", "scipy>=1.11.0"],
    },
)
