import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
SRC = os.path.join(RAW, "goyal_1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ.xlsx")

def main():
    df = pd.read_excel(SRC, sheet_name="Monthly")
    print("columns:", list(df.columns))
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"yyyymm": "yyyymm"})
    df["yyyymm"] = df["yyyymm"].astype(int)
    df = df[(df["yyyymm"] >= 195612) & (df["yyyymm"] <= 201612)].copy()
    df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)

    dp = np.log(df["D12"]) - np.log(df["Index"])
    ep = np.log(df["E12"]) - np.log(df["Index"])
    macro = pd.DataFrame({
        "date": df["date"],
        "dp": dp, "ep": ep, "bm": df["b/m"], "ntis": df["ntis"],
        "tbl": df["tbl"], "tms": df["lty"] - df["tbl"],
        "dfy": df["BAA"] - df["AAA"], "svar": df["svar"],
    })
    macro = macro.dropna()
    macro.to_csv(os.path.join(RAW, "macro_gw.csv"), index=False)

    rf = df[["date", "Rfree"]].dropna().rename(columns={"Rfree": "rf"})
    rf.to_csv(os.path.join(RAW, "riskfree.csv"), index=False)
    print("macro rows:", len(macro), macro["date"].min(), "->", macro["date"].max())
    print("rf rows:", len(rf), rf["date"].min(), "->", rf["date"].max())

if __name__ == "__main__":
    main()
