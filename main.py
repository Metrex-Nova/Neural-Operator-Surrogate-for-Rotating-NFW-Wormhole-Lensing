import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Neural Operator Surrogate for Gravitational Lensing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--mode", required=True,
                        choices=["generate", "train_fno", "train_baseline", "evaluate"])
    parser.add_argument("--data_dir",   type=str,   default="data/")
    parser.add_argument("--device",     type=str,   default="auto")

    parser.add_argument("--n_samples",  type=int,   default=2000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--n_workers",  type=int,   default=4)

    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)

    parser.add_argument("--modes",      type=int,   default=24)
    parser.add_argument("--width",      type=int,   default=32)
    parser.add_argument("--n_layers",   type=int,   default=4)

    parser.add_argument("--hidden",     type=int,   default=512)

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=" * 60)
    print(f"MODE: {args.mode}")
    print("=" * 60)

    if args.mode == "generate":
        from data.generate_dataset import generate_dataset
        generate_dataset(n_samples=args.n_samples, seed=args.seed,
                         out_dir=args.data_dir, n_workers=args.n_workers)

    elif args.mode == "train_fno":
        from training.train_fno import train_fno
        train_fno(data_dir=args.data_dir, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr,
                  modes=args.modes, width=args.width, n_layers=args.n_layers,
                  device_str=args.device)

    elif args.mode == "train_baseline":
        from training.train_baseline import train_baseline
        train_baseline(data_dir=args.data_dir, epochs=args.epochs,
                       batch_size=args.batch_size, lr=args.lr,
                       hidden=args.hidden, n_layers=args.n_layers,
                       device_str=args.device)

    elif args.mode == "evaluate":
        from evaluation.compare_models import compare_models
        compare_models(data_dir=args.data_dir, batch_size=args.batch_size,
                       device_str=args.device)


if __name__ == "__main__":
    main()
