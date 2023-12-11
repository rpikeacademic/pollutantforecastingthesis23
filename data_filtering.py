import pandas as pd

from typing import Union, Callable, Any

from city_neighbor import neighbor_by_distance, neighbor_by_wind_dir, neighbor_by_distance_target_specified

_PLACEHOLDER_CITYNAME = 'need_name'
_EXPORT_RESULT_TO_FILE = True
_X_FEATURE_LIST = ["SO2", "NO2", "PM10", "CO", "O3_8"]
_Y_FEATURE = "PM25"

## tau is the number of samples, i.e. the sequence length
TAU_NUM_SAMPLES = 7  # A previous week of samples (amount of observations)



def stdize_and_daysdelta_dataframe(df: pd.DataFrame, export_result: bool = _EXPORT_RESULT_TO_FILE,
                      _print_debug: bool = True) -> tuple[pd.DataFrame,Callable,tuple[float,float]]:
    '''
    Standardizes the feature columns in dataframe, and adds the daysdelta column.
    Returns: ( dataframe, undo_standardize_pm25 function, (pm25's std, pm25's mean) )
    '''

    # df: pd.DataFrame = pd.read_csv('jingjinji.csv')

    print('CSV DATAFRAME:')
    print(df)

    if _print_debug:
        def print_debug(*args, **kwargs):
            print(*args, *kwargs)
    else:
        def print_debug(*args, **kwargs):
            pass
        

    numeric_features = ["SO2", "NO2", "PM10", "CO", "O3_8", "PM25", "平均气压", "日照时数", "最高气温", "最小相对湿度", "year", "month", "season", "longitude", "latitude", "day"]

    x_features = ["SO2", "NO2", "PM10", "CO", "O3_8", "平均气压", "日照时数", "最高气温", "最小相对湿度"]
    y_feature = "PM25"


    _standardized_pm25_std = None
    _standardized_pm25_mean = None
    def undo_standardize_pm25(standardized_pm25):
            '''undo standardization for pm2.5 values (within calculation precision)
            \nWhen stardardizing: `st = (unst - mean) / std`
            therefore: `unst = (st * std) + mean`'''
            return (standardized_pm25 * _standardized_pm25_std) + _standardized_pm25_mean



    ##  CONVERT DATES to DAY NUMBERS

    # parse date column to a datetime value
    df['date'] = pd.to_datetime(df['date'], errors='raise', dayfirst=False)

    # delta number of days into an integer, for indexing/sorting like a timeseries
    print_debug('do day delta:')
    # _daydelta_start = pd.Timestamp(year=2000, month=1, day=1)  # start of year 2000
    _daydelta_start = df['date'][0]  # first entry's day
    print_debug('day delta start:', _daydelta_start)
    df['daysdelta'] = (df['date'] - _daydelta_start).dt.days  # `.dt.days` converts to integer days



    ##  STANDARDIZE used features

    # standardize
    to_standardize = x_features + [y_feature]
    feature_means = dict()
    feature_stds = dict()
    for feature in to_standardize:
        _mean = df[feature].mean()
        _std = df[feature].std()
        feature_means[feature] = _mean
        feature_stds[feature] = _std
        print_debug(f'feature: {feature}, mean: {_mean}, std: {_std}')
        df[feature] = (df[feature] - _mean) / _std
    print_debug('standardized features (hiding non-feature cols except city & daysdelta):')
    print_debug(df[ ['cityname'] + to_standardize + ['daysdelta'] ])

    # save pm2.5 (predicted feature) mean and std for standardization reversal later
    _standardized_pm25_mean = feature_means[y_feature]
    _standardized_pm25_std = feature_stds[y_feature]

    print_debug('first standardized PM25:', df[y_feature][0])
    print_debug('test undone standardization of first PM25:', undo_standardize_pm25(df[y_feature][0]))


    if export_result:
        df.to_csv('stdized_jingjinji.csv', encoding='utf-8', index=False)
        import json
        with open('stdized_pm25_info.json', 'w') as f:
            json.dump({'pm25_mean':_standardized_pm25_mean,'pm25_std':_standardized_pm25_std}, f)
        with open('stdized_infos.json', 'w') as f:
            json.dump({'means': feature_means, 'stds': feature_stds}, f)

    return df, undo_standardize_pm25, (_standardized_pm25_std, _standardized_pm25_mean)


# ---------------------------------------------------------
# ---------------------------------------------------------


def combine_dataframe_nearest_neighbor(df: pd.DataFrame, export_result: bool = _EXPORT_RESULT_TO_FILE,
                      _print_debug: bool = True) -> tuple[pd.DataFrame,Callable,tuple[float,float]]:
    '''
    Combine the nearest neighbor into each row in the dataframe
    Returns: the modified dataframe
    '''

    if _print_debug:
        def print_debug(*args, **kwargs):
            print(*args, *kwargs)
    else:
        def print_debug(*args, **kwargs):
            pass


    ##  DETERMINE NEIGHBOR for EACH ROW


    x_features = ["SO2", "NO2", "PM10", "CO", "O3_8", "平均气压", "日照时数", "最高气温", "最小相对湿度"]
    y_feature = "PM25"


    features = x_features + [y_feature]
    neighbor_prefix = 'neighbor_'

    # cache the cities' latitude and longitude
    city_latlong_dict: dict[str,tuple[float,float]] = {
        city: (lat,long)
        for (city, lat, long) in df[['cityname','latitude','longitude']].itertuples(index=False, name=None)
    }
    print_debug('first 3 key-val tuples out of city_latlong_dict:  ', list(city_latlong_dict.items())[:3])

    ## cache the cities' neighbors, by geo distance. This does not change from row to row.

    from city_neighbor import neighbor_by_distance_target_specified as neighbor_fn

    _latlongs = city_latlong_dict.items()

    city_nearest_dict = {
        city: neighbor_fn(city_latlong_dict, city, latlong)
        for city, latlong in _latlongs
    }
    print_debug('first 3 key-val tuples out of city_nearest_dict:  ', list(city_nearest_dict.items())[:3])



    ##  CACHE each city's row INDEXES for EACH DAYSDELTA timestamp
    ##  (essentially a reverse lookup table for indexing later)

    cities = list(set(df['cityname']))
    print_debug('len(cities), expecting 108: ', len(cities))

    #         like   dict[ city : dict[ daysdelta : row_index ] ]
    city_daysdelta_rowindex_dict: dict[str, dict[int, int]] = {
        city: dict() for city in cities
    }
    # {
    #     city: {
    #         daysdelta: (
    #             # row index for same daysdelta for neighboring city
    #             /
    #         )
    #         for 
    #     }
    #     for city in cities
    # }
    for ri, row in df.iterrows():
        city = row['cityname']
        daysdelta = row['daysdelta']
        # neighbor_city = city_nearest_dict[city]
        city_daysdelta_rowindex_dict[city][daysdelta] = ri



    # add new placeholding column for the neighbor city name
    df[neighbor_prefix + 'cityname'] = _PLACEHOLDER_CITYNAME
    # add new placeholding columns for the features
    for feature in features:
        df[neighbor_prefix + feature] = float('nan')

    print_debug('dataframe:')
    print_debug(df)
    # exit()


    ## COMBINE neighbors data onto the rows


    # for each row, determine neighbor city, and add its values into the placeholding rows
    _misses = 0
    _misses_inds = []
    for ri, row in df.iterrows():
        # row = df.iloc[ri]
        city = row['cityname']
        daysdelta = row['daysdelta']
        neighbor_city = city_nearest_dict[city]
        # if the neighbor does not have data for the same daysdelta, skip this row
        if daysdelta not in city_daysdelta_rowindex_dict[neighbor_city]:
            _misses += 1
            _misses_inds.append(ri)
            print_debug(f'  (#{_misses}) missing neighbor data for: city {city}, neighbor {neighbor_city}, daysdelta {daysdelta}  (ri {ri})')
            continue
        # reverse lookup the other row by index
        neighbor_row = df.iloc[ city_daysdelta_rowindex_dict[neighbor_city][daysdelta] ]

        # copy the neighbor's corresponding values over into the placeholder columns
        for feature in features:
            # like [row_index, column_name]
            df.loc[ri, neighbor_prefix + feature] = neighbor_row[feature]
        #
        df.loc[ri, neighbor_prefix+'cityname'] = neighbor_city

    # remove all the rows that
    df = df.drop(_misses_inds)
    print_debug(f'after drop of {len(_misses_inds)} rows:')
    print_debug(df)


    # # put the neighbor feature list into dataframe as a column
    # for feature in features:
    #     df['neighbor_'+feature] = dict(feature)


    print_debug('dataframe:')
    print_debug(df[['cityname', 'daysdelta'] + features + ['neighbor_cityname'] + [neighbor_prefix + f for f in features]])

    if export_result:
        df.to_csv('filtered_jingjinji.csv', encoding='utf-8', index=False)
        import json
        # with open('filtered_pm25_stdized_info.json', 'w') as f:
        #     json.dump({'pm25_mean':_standardized_pm25_mean,'pm25_std':_standardized_pm25_std}, f)
        with open('filtered_city_neighbors_cache.json', 'w') as f:
            json.dump(city_nearest_dict, f)

    return df#, undo_standardize_pm25, (_standardized_pm25_std, _standardized_pm25_mean)


# ---------------------------------------------------------
# ---------------------------------------------------------

import numpy as np


def sequence_bycity_padded_no_neighbors_dataframe(df: pd.DataFrame,
                                    x_features: list[str] = _X_FEATURE_LIST, y_feature: str = _Y_FEATURE,
                                    tau: int = TAU_NUM_SAMPLES,
                                    export_result: bool = _EXPORT_RESULT_TO_FILE,
                                    export_city_seq_folder = 'city_padded_seqs_noneighbors/',
                                    pad_amt_when_nodata = 1,
                                    _print_debug: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    '''
    Pad each city out with default values for missing rows (daysdeltas)
    so afterwards, each city will have a row for each daysdelta, although it may be an invalid row
    NOTE: due to padding operation, this operation will change any columns of dtype int into dtype float
    Returns a dict[ cityname , (X, Y) sequences of data for city ]
    '''
    
    assert 'daysdelta' in df.columns, 'dataframe df is expected to have a column \'daysdelta\' already generated'

    assert 'neighbor_cityname' not in df.columns, 'dataframe df is expected to NOT have neighbor information baked in'

    all_cities = sorted(set(df['cityname']))
    all_x_features = x_features

    if _print_debug:
        def print_debug(*args, **kwargs):
            print(*args, *kwargs)
    else:
        def print_debug(*args, **kwargs):
            pass

    daysdelta_range = range(df.daysdelta.min(), df.daysdelta.max() + 1)

    # assert df.daysdelta.min() == 0, 'daysdelta expected to start at 0'

    # for each daysdelta, which cities have a row for that daysdelta?
    daysdelta_city_dict: dict[int, set[str]] = {}

    columns_to_specify = ['cityname', 'daysdelta']
    columns_to_NaN = list( set(df.columns) - set(columns_to_specify) - {'date'} )  # - date will default it to NaT

    _consecutive_pads = 0

    for daysdelta in daysdelta_range:
        # which rows have a daysdelta matching this one
        mask = (df['daysdelta'].values == daysdelta)
        # get all cities who match the mask above (have data for this daysdelta)
        cities_with_data = set(df[mask]['cityname'])

        _stop_padding = (_consecutive_pads >= pad_amt_when_nodata)
        if len(cities_with_data) == 0:
            # reached a daysdelta with no data. pad?
            print_debug(f'no cities had data for daysdelta {daysdelta};'
                        f' {"skipping" if _stop_padding else "padding all cities with defaults"}!')
            if _stop_padding: # skip
                continue
            _consecutive_pads += 1  # going to be padding an entire no-data daysdelta
        else:
            _consecutive_pads = 0  # reached some amount of data again; reset pad count

        # record which cities had data for daysdelta
        daysdelta_city_dict[daysdelta] = cities_with_data

        # pad fake data row for the cities that did not
        _cities_without_data = set(all_cities) - cities_with_data
        if len(_cities_without_data) > 0:
           print_debug(f'data missing for daysdelta {daysdelta} for {len(_cities_without_data)}/{len(all_cities)} cities')
        for city in _cities_without_data:

            ## PAD THE CITIES THAT DO NOT HAVE DAYSDELTA (enter a NaN row, NaT for time)
            # by adding a new row at the end of the dataframe
            #             ['cityname', 'daysdelta'] + columns_to_NaN
            _row_to_add = [city,       daysdelta  ] + [float('nan')] * len(columns_to_NaN)
            df.loc[df.index.max() + 1, columns_to_specify + columns_to_NaN] = _row_to_add

    # end for daysdelta

    print_debug('df after padding:', df)

    # assert no new city names accidentally got introduced
    assert len(set(df['cityname']) - set(all_cities)) == 0, 'new city names somehow got introduced'



    ## SEQUENCE and EXPORT

    all_expected_daysdeltas = np.array( sorted(set(df['daysdelta'])) )
    print('len(all_expected_daysdeltas)', len(all_expected_daysdeltas))
    print('all_expected_daysdeltas.max()', all_expected_daysdeltas.max())

    TO_PRINT_LATER = []
    TO_PRINT_LATER.append(f'len(all_expected_daysdeltas): {len(all_expected_daysdeltas)}; all_expected_daysdeltas.max(): {all_expected_daysdeltas.max()}')

    city_sequences: dict[str, tuple[np.ndarray, np.ndarray]] = dict()

    df_groupByCity = df.groupby('cityname')

    _loopnumber = 0
    for city, _subset_object in df_groupByCity:
        print_debug()
        print_debug()
        print_debug('city:', city)
        print_debug(_subset_object)
        # sort by days delta:
        _subset_object = _subset_object.sort_values(by='daysdelta')
        print_debug('sorted by daysdelta:')
        print_debug(_subset_object)

        rows_x_ndarray = _subset_object[all_x_features].to_numpy()
        rows_y_ndarray = _subset_object[[y_feature]].to_numpy()
        print_debug()
        print_debug(rows_x_ndarray[:4])
        print_debug()
        print_debug(rows_y_ndarray[:4])
        # exit()
        print_debug()
        print_debug('rows_x_ndarray.shape:', rows_x_ndarray.shape)
        print_debug('rows_y_ndarray.shape:', rows_y_ndarray.shape)

        ## SPLIT INTO SEQUENCES

        # some prep adapted from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
        # multivariate section

        # will contain sequences
        X_data = []
        Y_data = []

        # note: 1-dimensional
        daysdelta = _subset_object['daysdelta'].to_numpy()

        print_debug()
        print_debug(city, 'len(daysdelta)', len(daysdelta))
        print_debug(city, 'daysdelta.max()', daysdelta.max())

        if(len(daysdelta) != len(all_expected_daysdeltas) or not (daysdelta == all_expected_daysdeltas).all()):
            print_debug(city, 'unexpected daysdelta not like all_expected_daysdeltas')
            TO_PRINT_LATER.append(f'{city} len(daysdelta): {len(daysdelta)}; daysdelta.max(): {daysdelta.max()}'
                                  f'; missing values: {set(all_expected_daysdeltas) - set(daysdelta)}')

        assert len(daysdelta) == rows_x_ndarray.shape[0] == rows_y_ndarray.shape[0]

        ## create a sequence of tau steps long for each possible VALID grouping
        for i, (deltai) in enumerate(daysdelta):
            if i+tau > len(daysdelta):  # no more sequences
                break
            if i==0: print_debug('i:', i, ', day delta:', deltai)

            # check sequence's validity (if all the daydeltas are sequential for slice [i:i+tau]):
            if not ((np.arange(0,tau) + deltai) == daysdelta[i:i+tau]).all():
                # not valid. a daysdelta is missing;
                #   either this city or neighbor city didn't have data for one of the daysdelta.
                print_debug(f'(city {city}) sequence daysdelta[{i}:{i+tau}] not valid consecutive daysdeltas ({list(daysdelta[i:i+tau])})')
                continue  # skip.

            if i==0: print_debug('passed daysdelta validity check')
            _x_seq = rows_x_ndarray[i:i+tau]
            _y_seq = rows_y_ndarray[i+tau-1]  # get only the last one
            if i==0: print_debug('x rows at [i:i+tau]:', _x_seq)
            if i==0: print_debug('y at [i+tau-1]:', _y_seq)
            X_data.append(_x_seq)
            Y_data.append(_y_seq)
        # end for ... in enumerate(daysdelta)
        
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)

        # X_data and Y_data now contain the sequences.

        print_debug('X_data.shape:', X_data.shape)
        print_debug('Y_data.shape:', Y_data.shape)
        
        city_sequences[city] = (X_data,Y_data)

        ## EXPORT to file
        if export_result:
            print_debug('exporting sequences for city', city, '...', end='')
            import pickle
            with open(export_city_seq_folder + city+'_padded_x.pickle', 'wb') as f:
                pickle.dump(X_data, f)
            with open(export_city_seq_folder + city+'_padded_y.pickle', 'wb') as f:
                pickle.dump(Y_data, f)
            print_debug('done')

        if _loopnumber == 0:
            _pdbackup = print_debug
            print_debug('debug: not printing in the remaining loops to avoid longer output')
            def print_debug(*args, **kwargs):
                pass
        _loopnumber += 1
    print_debug = _pdbackup

    # end for ... in df_groupByCity

    print_debug()
    print_debug('\n'.join(TO_PRINT_LATER))
    print_debug()

    print_debug('first city\'s [X, y] shapes:', [ a.shape for a in next(iter(city_sequences.values())) ] )

    _shapes = [(city, X.shape, y.shape) for city, (X, y) in city_sequences.items()]
    print_debug('_shapes[0:4]:', _shapes[0:4])

    assert all( ((_shapes[0][1] == s[1]) and (_shapes[0][2] == s[2])) for s in _shapes ), \
        'X and y should have the same shapes for all cities! same number of sequences and y vals'

    assert set(all_cities) == set(cityname for cityname in city_sequences), \
        'All cities should be present in the final city_sequences dict!'


    if export_result:
        print_debug('exporting entire sequence dict ...', end='')
        import pickle
        with open('city_padded_seqs_noneighbors_dict.pickle', 'wb') as f:
            pickle.dump(city_sequences, f)
        print_debug('done')

    return city_sequences


# ---------------------------------------------------------
# ---------------------------------------------------------


import pandas as pd
import numpy as np
from typing import Union

def sequence_bycity_dataframe(df: pd.DataFrame, x_features: list[str] = _X_FEATURE_LIST, y_feature: str = _Y_FEATURE,
                    tau: int = TAU_NUM_SAMPLES,
                    neighbor_prefix: Union[None, str, list[str]] = 'neighbor_',
                    export_result: bool = _EXPORT_RESULT_TO_FILE,
                    export_city_seq_folder = 'city_seqs_prejoined/',
                    _print_debug: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    '''
    Returns a dict[ cityname , (X, Y) sequences of data for city ]
    '''

    cities = list(set(df['cityname']))

    if _print_debug:
        def print_debug(*args, **kwargs):
            print(*args, *kwargs)
    else:
        def print_debug(*args, **kwargs):
            pass

    #

    features = x_features + [y_feature]
    # neighbor_prefix = 'neighbor_'
    ## include neighbor's y_feature ('neighbor_PM25')
    
    assert neighbor_prefix is None or isinstance(neighbor_prefix, (str, list))
    if neighbor_prefix is None:
        neighbor_prefix = []
    if isinstance(neighbor_prefix, str):
        neighbor_prefix = [neighbor_prefix]

    all_x_features = x_features
    for _prefix in neighbor_prefix:
        # if neighbor_prefix was empty or none, no extra features will be generated
        all_x_features += [_prefix + feature for feature in features]

    print_debug('all_x_features: ', all_x_features)

    ##  FINALLY: take FEATURE ROWS into LISTS of PREVIOUS SAMPLES to use (tau long)
    #     (Checking that they are contiguous days, i.e. no skipped days)

    # group by city
    df_groupByCity = df.groupby('cityname')
    print_debug('grouped by city:')
    print_debug(df_groupByCity)

    city_sequences: dict[str, tuple[np.ndarray, np.ndarray]] = dict()

    # create X, Y for each and batch them
    for name, _subset_object in df_groupByCity:
        print_debug('city:', name)
        print_debug(_subset_object)
        # sort by days delta:
        print_debug('sorted by daysdelta:')
        print_debug(_subset_object.sort_values(by='daysdelta'))

        rows_x_ndarray = _subset_object[all_x_features].to_numpy()
        rows_y_ndarray = _subset_object[[y_feature]].to_numpy()
        print_debug()
        print_debug(rows_x_ndarray[:4])
        print_debug()
        print_debug(rows_y_ndarray[:4])
        # exit()
        print_debug()
        print_debug('rows_x_ndarray.shape:', rows_x_ndarray.shape)
        print_debug('rows_y_ndarray.shape:', rows_y_ndarray.shape)

        ## SPLIT INTO SEQUENCES

        # some prep adapted from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
        # multivariate section

        # will contain sequences
        X_data = []
        Y_data = []

        # note: 1-dimensional
        daysdelta = _subset_object['daysdelta'].to_numpy()

        print_debug('len(daysdelta):',len(daysdelta))
        assert len(daysdelta) == rows_x_ndarray.shape[0] == rows_y_ndarray.shape[0]

        ## create a sequence of tau steps long for each possible VALID grouping
        for i, (deltai) in enumerate(daysdelta):
            if i+tau > len(daysdelta):  # no more sequences
                break
            if i==0: print_debug('i:', i, ', day delta:', deltai)

            # check sequence's validity (if all the daydeltas are sequential for slice [i:i+tau]):
            if not ((np.arange(0,tau) + deltai) == daysdelta[i:i+tau]).all():
                # not valid. a daysdelta is missing;
                #   either this city or neighbor city didn't have data for one of the daysdelta.
                print_debug(f'(city {name}) sequence daysdelta[{i}:{i+tau}] not valid consecutive daysdeltas ({list(daysdelta[i:i+tau])})')
                continue  # skip.

            if i==0: print_debug('passed daysdelta validity check')
            _x_seq = rows_x_ndarray[i:i+tau]
            _y_seq = rows_y_ndarray[i+tau-1]  # get only the last one
            if i==0: print_debug('x rows at [i:i+tau]:', _x_seq)
            if i==0: print_debug('y at [i+tau-1]:', _y_seq)
            X_data.append(_x_seq)
            Y_data.append(_y_seq)
        # end for ... in enumerate(daysdelta)
        
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)

        # X_data and Y_data now contain the sequences.

        print_debug('X_data.shape:', X_data.shape)
        print_debug('Y_data.shape:', Y_data.shape)
        
        city_sequences[name] = (X_data,Y_data)

        if export_result:
            print_debug('exporting sequences for city', name, '...', end='')
            import pickle
            with open(export_city_seq_folder + name+'_x.pickle', 'wb') as f:
                pickle.dump(X_data, f)
            with open(export_city_seq_folder + name+'_y.pickle', 'wb') as f:
                pickle.dump(Y_data, f)
            print_debug('done')

        # exit()

    # end for ... in df_groupByCity

    if export_result:
        print_debug('exporting entire sequence dict ...', end='')
        import pickle
        with open('city_seq_prejoined_dict.pickle', 'wb') as f:
            pickle.dump(city_sequences, f)
        print_debug('done')

    return city_sequences




def get_tensorloader(presliced_samples_Xs, presliced_labels, train):# -> mx.gluon.data.DataLoader:
    raise RuntimeError("not needed anymore as it can be fed quickly into ArrayDataset via the jupyter notebook")

    ## BATCH the SEQUENCES

    # X_batches = []

    # get_tensorloader(self, tensors, train, indices=slice(0, None))

    # # argument `tensors` would be like [features, labels]
    # tensors = tuple(a[indices] for a in tensors)

    # dataset = mx.gluon.data.ArrayDataset(*tensors)
 
    # return mx.gluon.data.DataLoader(dataset, self.batch_size,
    #                                 shuffle=train)



def get_dataloader(bycity_sequences, training: bool = True,
                   batch_size: int = 8,
                   test_cities: Union[list[str],slice,tuple[slice]] = (slice(10,21),slice(31,41)),
                   train_cities: Union[list[str],slice,tuple[slice]] = None,
                   _print_debug: bool = True) -> Any:  # <<<<TODO set proper return type!!
    '''
    Training will shuffle if True, not shuffle if False.
    '''
    # validate and resolve some params first
    
    if test_cities is None and train_cities is None:
        raise ValueError('either test_cities or train_cities (or both) must be specified')

    cities = list(set(df['cityname']))

    # condense test_cities to a list[str]
    if isinstance(test_cities, slice):
        test_cities: list[str] = cities[test_cities]
    elif isinstance(test_cities, tuple):
        slices = test_cities
        test_cities: list[str] = []
        for sl in slices: test_cities += cities[sl]
    # condense test_cities to a list[str]
    if isinstance(train_cities, slice):
        train_cities: list[str] = cities[train_cities]
    elif isinstance(train_cities, tuple):
        slices = train_cities
        train_cities: list[str] = []
        for sl in slices: train_cities += cities[sl]
    # now one should definitely be a list[str]. Allow the other to be filled in if None
    if train_cities is None:
        train_cities: list[str] = [city for city in cities if city not in test_cities]
    if test_cities is None:
        test_cities: list[str] = [city for city in cities if city not in train_cities]
    
    if _print_debug:
        def print_debug(*args, **kwargs):
            print(*args, *kwargs)
    else:
        def print_debug(*args, **kwargs):
            pass

    print_debug('train_cities: ', train_cities)
    print_debug('test_cities: ', test_cities)

    raise RuntimeError("not needed anymore as it can be fed quickly into ArrayDataset via the jupyter notebook")



if __name__ == '__main__':
    df: pd.DataFrame = pd.read_csv('jingjinji.csv')

    df, undo_pm25, (pm25_std, pm25_mean) = stdize_and_daysdelta_dataframe(df)

    city_XYseq_dict = sequence_bycity_padded_no_neighbors_dataframe(df, pad_amt_when_nodata=1)
    exit()

    # df = combine_dataframe_nearest_neighbor(df, )

    # df, undo_pm25, (pm25_std, pm25_mean) = prepare_dataframe(df)
    # read from cached csv
    df = pd.read_csv('filtered_jingjinji.csv')
    
    # seqs = sequence_bycity_dataframe(df)
    # seqs = sequence_bycity_dataframe(df, export_result=False)
    # read seqs from cached pickle
    import pickle
    with open('city_seq_dict.pickle', 'rb') as f:
        seqs = pickle.load(f)
    
    print()
    print(type(seqs), len(seqs))
    print(type(seqs['三门峡市']), len(seqs['三门峡市']))
    print(type(seqs['三门峡市'][0]), seqs['三门峡市'][0].shape)
    print(type(seqs['三门峡市'][1]), seqs['三门峡市'][1].shape)
    print('first 3 of X seqs for city "三门峡市":')
    print(seqs['三门峡市'][0][:3])
    print('first 3 of Y seqs for city "三门峡市":')
    print(seqs['三门峡市'][1][:3])
    exit()
    print()
    dataloader = get_dataloader(data, training=False)
    print()
    print('first of dataloader:')
    print(iter(dataloader))
