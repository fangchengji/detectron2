"""Downloads images from an item_profile csv data file.

Link to lomus query: https://lumos.idata.shopee.com/superset/sqllab?savedQueryId=60928

Requires pandas.
"""
import argparse
import urllib

import pandas as pd


####################
# HELPER_FUNCTIONS #
####################
def tidy_split(df, column, sep="|", keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df


def preprocess_item_profile(file_path: str) -> pd.DataFrame:
    """Takes in csv file path, and splits the images column,
    making each observation a single image url.
    """
    df = pd.read_csv(file_path)
    # Split the images column
    df = tidy_split(df, "images", sep=",")

    def form_url(row):
        country = row["country"].lower()
        if country in ["th", "id"]:
            country = "co." + country
        elif country in ["my", "br"]:
            country = "com." + country

        img = row["images"]

        return f"https://cf.shopee.{country}/file/{img}"

    # Form the full url
    df["img_url"] = df.agg(form_url, axis=1)

    return df


########
# MAIN #
########
def main(args):
    df = preprocess_item_profile(args.data_file)

    print(f"{len(df)} images to download.")

    # Download image files
    for _, row in df.iterrows():
        url = row["img_url"]
        name = url.split("/")[-1]
        country = row["country"].lower()
        output_path = f"{args.output}/{country}_{name}.jpg"

        try:
            urllib.request.urlretrieve(url, output_path)
        except Exception as e:
            print(e)
            print(f"Error for {url}")
        else:
            print(f"Downloaded {output_path}")

    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download images from item_profile csv data"
    )

    parser.add_argument("--data-file", help="path to csv file")
    parser.add_argument(
        "--output", help="path to output directory to store downloaded images"
    )
    args = parser.parse_args()

    main(args)
