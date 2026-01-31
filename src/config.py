import os
import torch
from dataclasses import dataclass

@dataclass
class SystemConfig:
    # Auto-detect environment
    IS_COLAB: bool = "COLAB_GPU" in os.environ

    # Paths
    @property
    def ROOT_DIR(self):
        if self.IS_COLAB:
            from google.colab import drive
            if not os.path.exists("/content/drive"):
                drive.mount("/content/drive")
            return "/content/drive/MyDrive/HELLoRA_Experiments"
        else:
            # RunPod / Local
            return "/workspace/HELLoRA_Experiments" if os.path.exists("/workspace") else "./HELLoRA_Experiments"

    def get_output_dir(self, subdir: str):
        path = os.path.join(self.ROOT_DIR, subdir)
        os.makedirs(path, exist_ok=True)
        return path

@dataclass
class TrainConfig:
    model_id: str = "allenai/OLMoE-1B-7B-0924"
    seed: int = 123
    max_len: int = 2048

    # Optimized for A6000/A100 (Effective Batch = 128)
    per_device_bs: int = 8
    grad_acc: int = 16

    # Paper Hyperparams
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05

    # Flags
    use_wandb: bool = True
    wandb_project: str = "hellora-repro"

SYS_CFG = SystemConfig()
TRAIN_CFG = TrainConfig()
