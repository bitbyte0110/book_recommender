import sys
import os

# Ensure local modules are importable
sys.path.append('src')
sys.path.append('.')

from evaluation.metrics import RecommenderEvaluator


def main():
    # Allow passing alpha values as command-line args, default to [0.0]
    if len(sys.argv) > 1:
        try:
            alpha_values = [float(a) for a in sys.argv[1:]]
        except ValueError:
            print("Invalid alpha value provided. Use numbers like: 0.0 0.5 1.0")
            sys.exit(1)
    else:
        alpha_values = [0.0]

    try:
        evaluator = RecommenderEvaluator(data_dir="data", models_dir="models")
        results = evaluator.run_full_evaluation(alpha_values=alpha_values)

        for alpha in alpha_values:
            m = results[alpha]
            print(
                f"Alpha {alpha:.1f}: F1={m['f1']:.4f}, RMSE={m['rmse']:.4f}, "
                f"Coverage={m['coverage']:.4f}, Diversity={m['diversity']:.4f}"
            )
    except Exception as e:
        print("Error during quick evaluation:", e)
        raise


if __name__ == "__main__":
    main()


