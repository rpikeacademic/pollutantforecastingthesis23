import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from typing import Union, Any

STD_COLS = ['PM25', 'PM10', 'CO', 'NO2', 'O3_8', 'SO2']

def standardize_dataframe_cols(df: pd.DataFrame,
                             cols: list[str] = STD_COLS
                             ) -> tuple[pd.DataFrame, dict[str,tuple[float,float]]]:
    '''
    Will perform standardization `col = (col - mean) / std`
    for each column in cols in the dataframe df.
    Transformation is performed in-place (meaning the transformed columns
    are placed back into the table).

    Returns: the DataFrame, and a dict containing tuples
    of the mean and std. dev. for each column in cols respectively
    '''
    means_stds = dict()
    for col in cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        # center mean to zero and then scale std to one, standardizing it
        df[col] = (df[col] - col_mean) / col_std
        means_stds[col] = (col_mean, col_std)

    return df, means_stds


def reverse_standardization(val: Union[float,int,list,tuple,np.ndarray,pd.DataFrame],
                          dict_means_stds: dict[str,tuple[float,float]],
                          dict_col = 'PM25',
                          df_col = None,
                          df_inplace = False
                          ):
    '''
    Will transform val to reverse its standardization according to the
    dict of means and std. dev.s, and the dict_col specified.
    Will retrieve the mean and standard deviation from the dict
    corresponding to the specified dict_col parameter.
    
    If val is a `pd.Dataframe`, df_col and df_inplace are used,
    for determining which column in the datatable to get and transform (df_col),
    and whether it should be replaced into the dataframe (df_inplace).
    When df_col is None, dict_col is used as the column name instead.
    Otherwise they are ignored.

    Input val types and their return types:
        - `float` or `int`:  `float`
        - `list`:            `list`
        - `tuple`:           `tuple`
        - `np.ndarray`:      `np.ndarray`
        - `pd.DataFrame`:    `pd.Series`
    '''
    mean, std = dict_means_stds[dict_col]
    if isinstance(val, (int,float)):
        return (val * std) + mean
    if isinstance(val, np.ndarray):
        return (val * std) + mean
    if isinstance(val, pd.DataFrame):
        _transformed_series = (val[dict_col if df_col is None else df_col] * std) + mean
        if df_inplace:
            # set the new column back in place
            val[dict_col] = _transformed_series
        return _transformed_series
    if isinstance(val, (list,tuple)):
        return type(val)((v * std) + mean for v in val)
    raise TypeError(f"unexpected type {type(val)} for parameter val!")


def normalize_wind_speed(df: pd.DataFrame, col: str
                         ) -> tuple[pd.DataFrame, Any, Any]:
    '''
    Linearly scales the column col in df
    to within [-1, 1], scaling centered at zero.
    
    For example, column values of `[-4, 2, 0, 10]`
    become `[-0.4, 0.2, 0, 1]`;
        and the windspeed min and max (original) would be `-4` and `10`, respectively.

    Returns df, the windspeed min (original), the windspeed max (original)
    '''
    windspeed_norm_min = df[col].min()
    windspeed_norm_max = df[col].max()
    largest_in_magnitude = max(abs(windspeed_norm_min), abs(windspeed_norm_max))
    scalefactor = 1 / largest_in_magnitude
    df[col] = df[col] * scalefactor
    return df, windspeed_norm_min, windspeed_norm_max


def reverse_normalized_wind_speed(val: Union[float,int,list,tuple,np.ndarray,pd.Series,pd.DataFrame],
                                  windspeed_norm_min_original,
                                  windspeed_norm_max_original,
                                  df_col = None,
                                  in_place = False):
    '''
    Will transform val to reverse its normalization according to the
    windspeed min and max specified.
    Will determine the rescale factor based on the norm's original min and max.

    Rescales the val -- or vals, if an array -- to fit inside
    [windspeed_norm_min_original, windspeed_norm_max_original].
    
    df_col: If val's type is a `pd.DataFrame`, df_col is used to know
    which column to apply reverse_normalized_wind_speed to.

    in_place: If val's type is a `list`, `np.ndarray`, or `pd.DataFrame`,
    then the in_place argument will be used.
    Otherwise it is ignored.

    Input val types and their return types:
        - `float` or `int`:  `float`
        - `list`:            `list`
        - `tuple`:           `tuple`
        - `np.ndarray`:      `np.ndarray`
        - `pd.Series`:       `pd.Series`
        - `pd.DataFrame`:    `pd.Series`
    '''
    ogmin = windspeed_norm_min_original
    ogmax = windspeed_norm_max_original
    # normal scale factor is 1 / max,
    #   so the inverse is max ( == 1 / (1 / max) )
    inversescalefactor = max(abs(ogmin),abs(ogmax))

    if isinstance(val, (int,float)):
        return (val * inversescalefactor)
    if isinstance(val, (np.ndarray, pd.Series)):
        if in_place:
            for i in range(len(val)):
                val[i] = val[i] * inversescalefactor
            return val
        return (val * inversescalefactor)
    if isinstance(val, pd.DataFrame):
        val.multiply()
        _transformed_series = (val * inversescalefactor)
        if in_place:
            # set the new column back in place
            val[df_col] = _transformed_series
        return _transformed_series
    if isinstance(val, list):
        if in_place:
            for i in range(len(val)):
                val[i] = val[i] * inversescalefactor
            return val
        return list((v * inversescalefactor) for v in val)
    if isinstance(val, tuple):
        return tuple((v * inversescalefactor) for v in val)
    raise TypeError(f"unexpected type {type(val)} for parameter val!")


def max_abs_difference(v1: Union[float,int,list,tuple,np.ndarray,pd.Series],
                            v2: Union[float,int,list,tuple,np.ndarray,pd.Series],
                            _try_coerce_if_unexpected = False):
        '''
        Returns the difference of the values. If they are lists of some type,
        Return the maximum absolute difference between the lists element-wise.
        '''
        # if v1 is v2: return None  <-- not doing this shortcut, for return type expectancy purposes

        if type(v1) != type(v2):
            raise TypeError(f'type of v1 did not match type of v2 ({type(v2)} != {type(v1)})')
        # compare easy ones immediately
        if isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
            return abs(v2 - v1)

        # handle asserting same shape of those that are list-like
        if hasattr(v1, 'shape') and hasattr(v2, 'shape'):
            if not (v1.shape == v2.shape):
                raise ValueError(f'shapes did not match. {v1.shape} != {v2.shape}')
        elif hasattr(v1, '__len__') and hasattr(v2, '__len__'):
            if not (len(v1) == len(v2)):
                raise ValueError(f'lengths did not match. {len(v1)} != {len(v2)}')

        if isinstance(v1, (list,tuple)) and isinstance(v2, (list,tuple)):
            return max(abs(b - a) for a, b in zip(v1, v2))
        
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            return np.abs(v2 - v1).max()
        
        if isinstance(v1, pd.Series) and isinstance(v2, pd.Series):
            return (v2 - v1).abs().max()


        # try to generalize this function as much as possible
        
        if _try_coerce_if_unexpected:

            from warnings import warn
            warn(RuntimeWarning(f"The types of v1 and v2 were not one of the"
                                + f" expected types (float, int, list, tuple, np.ndarray, pd.Series)"
                                + f". Attempting to coerce a max abs difference anyway."))

            are_subtractable = hasattr(v1, '__sub__') and hasattr(v2, '__sub__')
            are_iterables = hasattr(v1, '__iter__') and hasattr(v2, '__iter__')
            are_indexed = hasattr(v1, '__getitem__') and hasattr(v2, '__getitem__')
            are_indexed_iterables = are_iterables and are_indexed
            
            if are_indexed_iterables:
                # treat them like lists or tuples
                return max(abs(b - a) for a, b in zip(v1, v2))
            
            if are_subtractable:
                # try to get the absolute of the difference
                # example: for sets this would run but not continue into abs(...)
                diff = (v2 - v1)
                if hasattr(diff, '__abs__'):
                    absdiff = abs(diff)
                    absdiff_is_iterable = hasattr(absdiff, '__iter__')
                    # get the max value if it's an iterable
                    if absdiff_is_iterable:
                        return max(absdiff)
                    # try to make it a float or int
                    if hasattr(absdiff, '__float__'):
                        return float(absdiff)
                    if hasattr(absdiff, '__int__'):
                        return int(absdiff)
                del diff
            
            # just try to coerce it
            if hasattr(v1, '__float__') and hasattr(v2, '__float__'):
                return abs(float(v2) - float(v1))
            if hasattr(v1, '__int__') and hasattr(v2, '__int__'):
                return abs(int(v2) - int(v1))

        raise RuntimeError(f"unable to determine the max abs difference between"
                           + f" v1 and v2.\n  v1: {v1} ; v2: {v2}"
                           + f"\n  type(v1): {type(v1)}, type(v2): {type(v2)}")
    


# test the reversal
if __name__ == '__main__':

    from data_utils_df_joiner import left_join_the_dataframes
    df, cities_names_dict, cities_latlongs_dict = left_join_the_dataframes()

    ogval = df.loc[0, 'WS.max']
    ogval_slice = df.loc[0:8, 'WS.max']
    df, ws_norm_min, ws_norm_max = normalize_wind_speed(df, 'WS.max')
    normedval = df.loc[0, 'WS.max']
    normedval_slice = df.loc[0:8, 'WS.max']  # pd.Series

    print('ogval:', ogval)
    print('ogval_slice:\n', ogval_slice)
    print('normedval:', normedval)
    print('normedval_slice:\n', normedval_slice)
    print('ws_norm_min:', ws_norm_min)
    print('ws_norm_max:', ws_norm_max)

    
    ## assertions on wind speed normalization reversal
    #  to verify the reversal is (nearly) accurate:
    
    allowed_error = 9e-16

    # single value
    reversed_norm_val = reverse_normalized_wind_speed(
        normedval, ws_norm_min, ws_norm_max)
    
    _error = max_abs_difference(ogval, reversed_norm_val)
    print(_error)
    assert _error < allowed_error, \
        f'error too large {_error}; {ogval} != {reversed_norm_val}'
    
    # slice passed in
    reversed_normedval_slice = reverse_normalized_wind_speed(
        normedval_slice, ws_norm_min, ws_norm_max)
    
    # comparing as Series objects
    _error = max_abs_difference(ogval_slice, reversed_normedval_slice)
    print(_error)
    assert _error < allowed_error, \
        f'error too large {_error}; {list(ogval_slice)} != {list(reversed_normedval_slice)}'
    # comparing as lists
    _error = max_abs_difference(list(ogval_slice), list(reversed_normedval_slice))
    print(_error)
    assert _error < allowed_error, \
        f'error too large {_error}; {list(ogval_slice)} != {list(reversed_normedval_slice)}'
    
    # list of slice passed in
    reversed_list_normedval_slice = reverse_normalized_wind_speed(
        list(normedval_slice), ws_norm_min, ws_norm_max)
    
    _error = max_abs_difference(list(ogval_slice), reversed_list_normedval_slice)
    print(_error)
    assert _error < allowed_error, \
        f'error too large {_error}; {list(ogval_slice)} != {reversed_list_normedval_slice}'
    
    # in-place list
    inplace_list = list(normedval_slice)
    reverse_normalized_wind_speed(
        inplace_list, ws_norm_min, ws_norm_max, in_place = True)
    
    _error = max_abs_difference(list(ogval_slice), inplace_list)
    print(_error)
    assert _error < allowed_error, \
        f'error too large {_error}; {list(ogval_slice)} != {inplace_list}'
    
    # numpy
    ogval_slice_np_copy = ogval_slice.to_numpy(copy=True)
    reversed_normedval_slice_numpy_copy = reverse_normalized_wind_speed(
        normedval_slice.to_numpy(copy=True), ws_norm_min, ws_norm_max)
    
    _error = max_abs_difference(ogval_slice_np_copy, reversed_normedval_slice_numpy_copy)
    print(_error)
    assert _error < allowed_error, \
        f'error too large {_error}; {ogval_slice_np_copy} != {reversed_normedval_slice_numpy_copy}'
    
    # in-place numpy
    normed_np_for_inplace = normedval_slice.to_numpy(copy=True)
    reverse_normalized_wind_speed(
        normed_np_for_inplace, ws_norm_min, ws_norm_max, in_place = True)
    
    _error = max_abs_difference(ogval_slice_np_copy, normed_np_for_inplace)
    print(_error)
    assert _error < allowed_error, \
        f'error too large {_error}; {ogval_slice_np_copy} != {normed_np_for_inplace}'
