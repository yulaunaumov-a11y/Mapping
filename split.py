import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    os.chdir(SCRIPT_DIR)
    train_dir = os.path.join(SCRIPT_DIR, "train")
    val_dir = os.path.join(SCRIPT_DIR, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train = sorted(
            f"train/{d}"
            for d in os.listdir(train_dir)
            if d.startswith("camera_door2_")
            and os.path.isdir(os.path.join(train_dir, d))
        )
        val = sorted(
            f"val/{d}"
            for d in os.listdir(val_dir)
            if d.startswith("camera_door2_")
            and os.path.isdir(os.path.join(val_dir, d))
        )
        split = {"train": train, "val": val}
    else:
        random.seed(42)
        sessions_with_coords = sorted(
            d
            for d in os.listdir(".")
            if d.startswith("camera_door2_")
            and (
                os.path.exists(os.path.join(d, "coords_top.json"))
                or os.path.exists(os.path.join(d, "coords_bottom.json"))
            )
        )
        random.shuffle(sessions_with_coords)
        n_train = int(len(sessions_with_coords) * 0.8)
        split = {
            "train": sessions_with_coords[:n_train],
            "val": sessions_with_coords[n_train:],
        }

    with open(os.path.join(SCRIPT_DIR, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    print(f"Train : {len(split['train'])}")
    print(f"Val   : {len(split['val'])}")


if __name__ == "__main__":
    main()
