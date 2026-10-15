import argparse
import pandas as pd
from src.modeling import train_evaluate, save_artifacts

def main():
    parser = argparse.ArgumentParser(description="Train membrane-protein classifiers.")
    parser.add_argument("--data", default="data/proteins.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    models, results, split_name = train_evaluate(df, args.test_size, args.random_state)
    save_artifacts(models, results, split_name)
    print("Split strategy:", split_name)
    for name, metrics in results.items():
        print("\n" + name)
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    print("\nSaved models and metrics in ./models/")

if __name__ == "__main__":
    main()
