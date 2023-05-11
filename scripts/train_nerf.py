import argparse
import os

def train(transforms_json: str):
    # run ns-train command
    os.system(f"ns-train nerfacto-big --data {transforms_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms_json", type=str, default="transforms.json")
    args = parser.parse_args()
    train(args.transforms_json)