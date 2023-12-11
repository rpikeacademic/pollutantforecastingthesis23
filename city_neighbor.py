import pandas as pd

# type alias for the city lat/long location cache
LatLongDict = dict[str,tuple[float,float]]

_DIST_ANGLE_WEIGHT = 2
_ANGLE_FACTOR_WEIGHT = 1


## Note: use   `neighbor_by_distance_target_specified(...)`  below because it does not rely on the pd.DataFrame


# def neighbor_by_wind_dir(df: pd.DataFrame, target_city: str, daysdelta, _w=_DIST_ANGLE_WEIGHT) -> str:
def neighbor_by_wind_dir(city_locs_dict: LatLongDict, df: pd.DataFrame, row, _w_d=_DIST_ANGLE_WEIGHT, _w_a=_ANGLE_FACTOR_WEIGHT) -> str:
    '''returns the city that should be considered
    the neighbor city for `target_city` at time `daysdelta`'''

    distance_factor = _w_d
    angle_factor = _w_a
    target_lat = df['latitude']
    target_long = df['longitude']
    target_wind_dir = row['wind']
    target_city = row['city']
    
    # return city with minimum angle over distance depending on the weight
    min_val = 1e6
    min_city: str = None
    for city, (lat, long) in city_locs_dict.items():
        if target_city == city:
            continue
        # find angle vs distance factor
        angle = geo_angle(target_lat, target_long, lat, long)
        dist = geo_distance_haversine(target_lat, target_long, lat, long)
        angle_dist_factor = angle * angle_factor * (dist * distance_factor)
        # compare to save
        if angle_dist_factor < min_val:
            min_val = angle_dist_factor
            min_city = city

    assert min_city is not None
    return min_city


def neighbor_by_distance(city_locs_dict: LatLongDict, df: pd.DataFrame, row) -> str:
    '''returns the city that should be considered
    the neighbor city in this row (for this target city and time `daysdelta`)
    \nUsing the distance between the two cities'''

    timedelta = row['daysdelta']
    target_lat = row['latitude']
    target_long = row['longitude']
    target_city = row['city']

    # return city with minimum distance from target city    
    min_val = 1e6
    min_city: str = None
    for city, (lat, long) in city_locs_dict.items():
        if target_city == city:
            continue
        dist = geo_distance_haversine(target_lat, target_long, lat, long)
        if dist < min_val:
            min_val = dist
            min_city = city

    assert min_city is not None
    return min_city


def neighbor_by_distance_target_specified(city_locs_dict: LatLongDict, target_city: str, target_lat_long: tuple[float, float]):
    '''returns the city (out of the dict of cities provided)
    that should be considered the neighbor city in this row
    \nUsing the distance between the two cities (latitude, longitude)'''
    target_lat, target_long = target_lat_long

    # return city with minimum distance from target city    
    min_val = 1e6
    min_city: str = None
    for city, (lat, long) in city_locs_dict.items():
        if target_city == city:
            continue
        dist = geo_distance_haversine(target_lat, target_long, lat, long)
        if dist < min_val:
            min_val = dist
            min_city = city

    assert min_city is not None
    return min_city


from math import cos, asin, sqrt

# Haversine distance, from Wikipedia, implemented by: https://stackoverflow.com/a/41337005/14390381
def geo_distance_haversine(lat1, long1, lat2, long2):
    p = 0.017453292519943295  # deg->radian conversion, pi/180
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((long2-long1)*p)) / 2
    return 12742 * asin(sqrt(hav))  # 12742 km diameter of Earth


def geo_angle(lat1, long1, lat2, long2):
    # for two points, gets the angle in degrees that the line between those two points is off of East (a line facing east)
    from shapely.geometry import Point
    P1 = Point(lat1, long1)
    interP = Point(lat2, long2)
    # a false point slightly to the east, to provide a second line that is "horizontal" along lat1
    P2 = Point(lat1, long1+1)
    # azimuth, Vincenty's formulae
    from pygc import great_distance  # distance in meters
    P2az = great_distance(start_latitude=P2.y, start_longitude=P2.x,
                          end_latitude=interP.y,end_longitude=interP.x)['reverse_azimuth']
    P1az = great_distance(start_latitude=P1.y, start_longitude=P1.x,
                          end_latitude=interP.y,end_longitude=interP.x)['reverse_azimuth']
    return P2az - P1az


