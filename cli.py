import argparse, os
from src.agent_lc.pipeline import run_pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/sample_transactions.csv")
    p.add_argument("--out", default="reports/risk_report.xlsx")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res = run_pipeline(args.csv, args.out)
    print(res)

if __name__ == "__main__":
    main()
