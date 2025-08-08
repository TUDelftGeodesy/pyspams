import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spams10_filepath",
        "-spams10_fp",
        default="data/nl_krimpenerwaard_spams10.parquet",
        help="str, path to the SPAMS parameters file",
    )
    parser.add_argument(
        "--meteo_dir",
        "-met_dir",
        default="data/",
        help="str, path to the meteo file",
    )
    parser.add_argument(
        "--meteo_filename",
        "-met_fn",
        help="str, the meteo data file name (optional). If specified, will use this station even though it is not the closest one",
    )
    parser.add_argument(
        "--start_date",
        "-sdate",
        default=20200101,
        help="int, start date when computing the SPAMS model",
    )
    parser.add_argument(
        "--end_date",
        "-edate",
        default=20241231,
        help="int, end date when computing the SPAMS model",
    )
    parser.add_argument(
        "--parcel_id",
        "-pid",
        help="int, parcel ID (optional). It will take a random parcel if it is empty",
    )

    args = parser.parse_args()

    ## Load SPAMS parameters
    df_spams = pd.read_parquet(args.spams10_filepath)
    pnt_ids = df_spams["pnt_id"].unique()

    ## Time series dates
    start = pd.to_datetime(datetime.strptime(str(args.start_date), "%Y%m%d"))
    end = pd.to_datetime(datetime.strptime(str(args.end_date), "%Y%m%d"))
    epoch = pd.date_range(start=start, end=end)

    ## Select a parcel
    if args.parcel_id is None:
        pid = np.random.choice(pnt_ids)
    else:
        pid = np.int32(args.parcel_id)
    df_spams_sel = df_spams[df_spams["pnt_id"] == pid]

    ## SPAMS model parameters of the selected parcel
    xP = df_spams_sel["xP"].values[0]
    xE = df_spams_sel["xE"].values[0]
    xI = df_spams_sel["xI"].values[0]
    t = int(df_spams_sel["tau"].values[0])

    ## SPAMS model parameters stdev of the selected parcel
    std_xP = np.sqrt(df_spams_sel["var_xP"].values[0])
    std_xE = np.sqrt(df_spams_sel["var_xE"].values[0])
    std_xI = np.sqrt(df_spams_sel["var_xI"].values[0])

    ## Load and subset meteo data
    if args.meteo_filename is None:
        filelist = sorted(Path(args.meteo_dir).glob("**/etmgeg_*.txt"))
        f1 = [None] * len(filelist)
        for i, file in enumerate(filelist):
            f1[i] = utils.read_knmi(file)
        df_meteo = pd.concat(f1)

        meteo_id = df_spams_sel["meteo_id"].values[0]

        meteo_subset = df_meteo.loc[
            (df_meteo["meteo_id"] == meteo_id)
            & (df_meteo["datum"] > (start - timedelta(days=t)))
            & (df_meteo["datum"] <= end)
        ].reset_index(drop=True)
    else:
        df_meteo = utils.read_knmi(os.path.join(args.meteo_dir, args.meteo_filename))

        meteo_subset = df_meteo.loc[
            (df_meteo["datum"] > (start - timedelta(days=t)))
            & (df_meteo["datum"] <= end)
        ].reset_index(drop=True)

    ## Compute SPAMS model and mean irreversible rate
    reversible, irreversible, height = utils.spams_model(xP, xE, xI, t, meteo_subset)
    vI, std_vI = utils.irreversible_rate(
        irreversible, xI, df_spams_sel["var_xI"].values[0]
    )

    ## Compute F-value
    F = utils.f_value(df_spams_sel["rss"].values[0], df_spams_sel['dof'].values[0])

    ## Plot for a parcel
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(epoch, height, "-k", markersize=2, label="Total")
    ax1.plot(epoch, reversible, "-.C7", markersize=2, label="Reversible")
    ax1.plot(epoch, irreversible, "--C7", linewidth=2, label="Irreversible")

    ax1.set_ylabel("Relative surface elevation (mm)", fontsize=9)
    ax1.grid(linestyle="--", alpha=0.5)
    ax1.set_ylim(ymin=min(height) - 50, ymax=max(height) + 50)

    ## Add information using text boxes
    info_text = (
        f"Parcel ID: {pid}\n"
        f"Loc (lon, lat): {round(df_spams_sel['pnt_lon'].values[0], 4)}, {round(df_spams_sel['pnt_lat'].values[0], 4)}\n"
        f"F-value = {round(F, 2)}\n"
        f"SPAMS parameters:\n"
        f"\t$x_P$ = {utils.format_with_uncertainty(xP, std_xP)}\n"
        f"\t$x_E$ = {utils.format_with_uncertainty(xE, std_xE)}\n"
        f"\t$x_I$ = {utils.format_with_uncertainty(xI, std_xI)} mm/day\n"
        f"\t$\\tau$ = {t}\n"
        f"Estimated yearly irreversible rate:\n"
        f"\t$\overline{{v_I}}$ = {utils.format_with_uncertainty(vI, std_vI)} mm/year"
    )
    ax1.text(
        0.01,
        0.86,
        info_text,
        verticalalignment="center",
        transform=ax1.transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    )

    if "krimpenerwaard" in args.spams10_filepath:
        source_text = "Source: https://doi.org/10.4121/dfbe9109-d058-4a64-a5b4-1cc9d9a5f836"
    elif "delfland" in args.spams10_filepath:
        source_text = "Source: https://doi.org/10.4121/c8646561-5475-4e23-bd90-3b1a3375ac7c"
    else:
        source_text = "Source: no DOI found"
    ax1.text(
        0.01,
        0.03,
        source_text,
        verticalalignment="center",
        transform=ax1.transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    )

    ## Meteorological data plot
    meteo_subset = df_meteo.loc[
        (df_meteo["datum"] > start) & (df_meteo["datum"] <= end)
    ].reset_index(drop=True)

    ax2 = ax1.twinx()
    ax2.bar(
        meteo_subset["datum"],
        meteo_subset["precip"],
        color="blue",
        label="Daily precip.",
    )
    ax2.bar(
        meteo_subset["datum"],
        -meteo_subset["evapo"],
        color="red",
        label="Daily evapo. (negative)",
    )

    ax2.set_ylabel("Daily precip. / evapo. amount (mm)", fontsize=9)
    ax2.set_ylim(
        ymin=max(meteo_subset["evapo"]) * -1 - 5,
        ymax=max(meteo_subset["precip"]) + 30,
    )

    fig.legend(bbox_to_anchor=(0.94, 0.97), fontsize=8)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
