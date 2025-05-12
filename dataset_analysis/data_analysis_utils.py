import pandas as pd
from urllib.parse import urlparse


class DsDetails:
    def __init__(self, dataset_name: str):
        if dataset_name == '12M':
            path = "hf://datasets/dclure/laion-aesthetics-12m-umap/train.parquet"
            cap_column = 'TEXT'
            url_column = 'URL'
            overfit_column = None
        elif dataset_name == '30k':
            path = 'prompt_lists/membership_attack_top30k.parquet'
            cap_column = 'caption'
            url_column = 'url'
            overfit_column = None
        elif dataset_name == 'sdv1':
            path = 'prompt_lists/sdv1_bb_edge_groundtruth.parquet'
            cap_column = 'caption'
            url_column = 'url'
            overfit_column = 'overfit_type'
        elif dataset_name == 'sdv2':
            path = 'prompt_lists/sdv2_bb_edge_groundtruth.parquet'
            cap_column = 'caption'
            url_column = 'url'
            overfit_column = 'overfit_type'
        else:
            raise ValueError('Dataset not supported')
        self.path = path
        self.cap_column = cap_column
        self.url_column = url_column
        self.overfit_column = overfit_column
        self.df = self.load_dataset()

    def load_dataset(self):
        df = pd.read_parquet(self.path, engine='pyarrow')
        return df

    def filter_by_idiom(self, idiom: str, cap_or_url: str, match_case: bool = False, inplace=False):
        """Search for a specific idiom either in the caption"""
        if cap_or_url == 'cap':
            col_name = self.cap_column
        elif cap_or_url == 'url':
            col_name = self.url_column
        else:
            raise ValueError('cap_or_url must be either cap or url')

        mask = self.df[[col_name]].apply(
            lambda x: x.str.contains(
                idiom,
                regex=True,
                case=match_case
            )
        ).any(axis=1)
        filtered = self.df.loc[mask]
        if inplace:
            self.df = filtered
        return filtered

    def filter_by_url_list(self, idiom: str, col_name: str):
        df = self.df[self.df[col_name].apply(
            lambda lst: any(idiom.lower() in s.lower() for s in lst))]
        return df

    def filter_by_overfit_type(self, overfit_type: str):
        if self.overfit_column is not None:
            mask = self.df[[self.overfit_column]].apply(
                lambda x: x.str.match(
                    overfit_type,
                )
            ).any(axis=1)
            self.df = self.df.loc[mask]
        else:
            raise ValueError('overfit_type is not specified for this dataset')

    def print_filtered_df(self, include_overfit_type: bool = False):
        if include_overfit_type:
            print(self.df.loc[:, [self.cap_column, self.url_column, self.overfit_column]])

        else:
            print(self.df.loc[:, [self.cap_column, self.url_column]])


def strip_website_name(url: str) -> str:
    return urlparse(url).netloc
