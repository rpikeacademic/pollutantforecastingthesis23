import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def left_join_the_dataframes(jingjinji_filepath = 'jingjinji.csv',
                        airquality_filepath = 'aq_.csv'
                        ) -> tuple[pd.DataFrame, dict[int,str], dict[int,tuple[float,float]]]:
    '''
    Joins the two csv dataframes.
    Left-join air quality and jingjinji, on citycode & date, which adds data from jingjinji.
    Averages the values of the shared pollutant columns ['PM25', 'PM10', 'CO', 'NO2',
       'O3_8', 'SO2'].
    Ensures each row has the correct cityname (from the jingjinji dataset)
    and latitude longitude coordinates, as left joining leaves some of that out.

    Returns the joined dataframe, the mapping for citycodes to citynames,
    and the mapping for citycodes to (latitude,longitude) locations
    '''

    df_j = pd.read_csv(jingjinji_filepath)
    df_aq = pd.read_csv(airquality_filepath)

    ## prep columns for joining

    df_aq = df_aq.rename(columns={"O3_8h": "O3_8", "PM2.5": "PM25"}, errors='raise')
    df_aq['date'] = pd.to_datetime(df_aq['date'], errors='raise', dayfirst=False)
    df_j['date'] = pd.to_datetime(df_j['date'], errors='raise', dayfirst=False)

    ## join on citycode (and date)

    df = pd.merge(
        df_aq, df_j, how='left', on=['citycode', 'date'],
    )

    citycodes_names_dict = dict()
    citycodes_latlongs_dict = dict()

    for cc in set(df_j.citycode):
        # perform mask once
        ccmask = df_j.citycode.values == cc
        df_j_masked = df_j[ccmask]

        # get a set of any and all name(s)
        # corresponding to the citycode cc
        citycodes_names_dict[cc] = set(df_j_masked['cityname'].values)
        # do the same for the latitude, longitude
        citycodes_latlongs_dict[cc] = set(
                zip(
                    df_j_masked['latitude'].values,
                    df_j_masked['longitude'].values
                )
            )
    
    # make sure this dict only has one name per cc!
    assert all(len(cnset) == 1 for cc, cnset in citycodes_names_dict.items())
    # make sure this dict only has one lat,long pair per cc!
    assert all(len(llset) == 1 for cc, llset in citycodes_latlongs_dict.items())
    
    # convert the dict to directly mapping
    # each citycode to its unique cityname
    citycodes_names_dict = {cc: list(cnset)[0]
                            for cc, cnset in citycodes_names_dict.items()}
    # do the same for the latitude, longitude
    citycodes_latlongs_dict = {cc: list(llset)[0]
                            for cc, llset in citycodes_latlongs_dict.items()}
    
    # ensure every row each citycode has its city name
    # (the column 'cityname_y', from 'jingjinji.csv' df_j)
    # and its lat-long coordinates,
    # because left joining would not fill that in.
    for cc, cname in citycodes_names_dict.items():
        ccmask = df.citycode.values == cc
        clat, clong = citycodes_latlongs_dict[cc]
        df.loc[ccmask,'cityname_y'] = cname
        df.loc[ccmask,'latitude'] = clat
        df.loc[ccmask,'longitude'] = clong
    assert len(set(df.cityname_y)) == len(set(df.latitude)) == len(set(df.longitude))

    # print(df[df['WS.max'].isna()])  # expecting 1 row with NaN in that column; 2023-03-25 for citycode 140100

    df['PM25'] = df[['PM25_x', 'PM25_y']].mean(axis=1)
    df['PM10'] = df[['PM10_x', 'PM10_y']].mean(axis=1)
    df['CO'] =   df[['CO_x',   'CO_y']].mean(axis=1)
    df['NO2'] =  df[['NO2_x',  'NO2_y']].mean(axis=1)
    df['O3_8'] = df[['O3_8_x', 'O3_8_y']].mean(axis=1)
    df['SO2'] =  df[['SO2_x',  'SO2_y']].mean(axis=1)

    # print(df[df['WS.max'].isna()])  # expecting 1 row with NaN in that column; 2023-03-25 for citycode 140100

    # df = df.sort_values(by='date')
    df = df.sort_values(by=['citycode','date'])

    # print(df[df['WS.max'].isna()])  # expecting 1 row with NaN in that column; 2023-03-25 for citycode 140100
    # exit()

    return df, citycodes_names_dict, citycodes_latlongs_dict


if __name__ == '__main__':
    df, cndict, clldict = left_join_the_dataframes()
    from datetime import datetime, timedelta

    oneday = timedelta(days=1)
    # date = datetime.fromtimestamp(df.date.min().timestamp())
    # maxdate = datetime.fromtimestamp(df.date.max().timestamp())
    date = df.date.min().to_pydatetime()
    maxdate = df.date.max().to_pydatetime()
    print(type(date))
    # dates = [print(npd,
    #                type(
    #                 datetime.utcfromtimestamp(
    #                     npd.item()))) \
    #          for npd in \
    #             set(df.date.values)]
    print(type(df.date.values[0]))
    print(type(df.date.values[0].item()))
    print(df.date.values)
    dates = [pd.Timestamp(npd.item()).to_pydatetime() for npd in set(df.date.values)]
    print(dates)
    print(type(dates[0]))
    
    while date <= maxdate:
        if date not in dates:
            print(date, 'not in dates')
        date = date + oneday


    # plt.scatter()
    df.plot.scatter('date','PM25',s=2)
    plt.show()