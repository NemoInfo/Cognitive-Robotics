import argparse

class DefaultConfig:
    def __init__(self):
        self.p = argparse.ArgumentParser()
        self.p.add_argument("--model-name", type=str, default="vision")
        self.p.add_argument("--val-ratio", type=float, default=0.2)
        self.p.add_argument("--num-epochs", type=int, default=50)
        self.p.add_argument("--batch-size", type=int, default=32)
        self.p.add_argument("--lr", type=float, default=1e-4)
        self.p.add_argument("--block-sizes", type=int, default=[32, 64, 128])

    def parse(self):
        return self.p.parse_args()
