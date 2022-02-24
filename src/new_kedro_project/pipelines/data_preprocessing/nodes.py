import pandas as pd
import numpy as np

from collections import defaultdict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def preprocess_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data for stroke prediction
    Args:
        stroke_data: Raw
    Returns:
        Preprocessed data, with categorical values encoded with OneHot and numerical values standardized.
    """
    y = data_df.stroke
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_df['bmi'] = imp_mean.fit_transform(data_df[['bmi']])
    train_ids = data_df.loc[:, 'id']
    train_cats = data_df.loc[:, data_df.dtypes == object]
    cat_cols = train_cats.columns

    encoder = OneHotEncoder(handle_unknown='ignore')
    oh_enc = encoder.fit_transform(train_cats).toarray()
    train_cats_enc = pd.DataFrame(oh_enc, columns=encoder.get_feature_names())
    final_cat_cols = list(train_cats_enc.columns)

    train_num = data_df.loc[:, data_df.dtypes != object].drop(columns=['stroke', 'id'])
    num_cols = train_num.columns
    standard_scaler = StandardScaler()
    train_num_std = standard_scaler.fit_transform(train_num)
    final_num_feats = list(num_cols)
    X = pd.DataFrame(np.hstack((train_cats_enc, train_num_std)), 
                     columns=list(final_cat_cols) + list(num_cols))
    final_df = pd.concat([X,y], axis=1)
    return final_df