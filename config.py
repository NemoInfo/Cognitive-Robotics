import argparse
from typing import List

class DefaultConfig:
    def __init__(self):
        self.p = argparse.ArgumentParser()
        self.p.add_argument("--model_name", type=str, default="vision")
        self.p.add_argument("--val_ratio", type=float, default=0.2)
        self.p.add_argument("--num_epochs", type=int, default=50)
        self.p.add_argument("--kernel_size", type=int, default=3)
        self.p.add_argument("--batch_size", type=int, default=32)
        self.p.add_argument("--lr", type=float, default=1e-4)
        self.p.add_argument("--blocks", nargs='+', type=int, default=[32, 64, 128])
        self.p.add_argument("--dropouts", nargs='+', type=float, default=[0.2, 0.2, 0.2])

    def parse(self):
        return self.p.parse_args()

    def args(self, **kwargs):
        args = []
        for key, val in kwargs.items():
            if isinstance(val, list):
                args.extend([f"--{key}", *map(str, val)])
            elif not isinstance(str, list):
                args.extend([f"--{key}", str(val)])
            else:
                args.extend([f"--{key}", val])

        return self.p.parse_args(args)
