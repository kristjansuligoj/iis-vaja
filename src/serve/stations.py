stations = [
        {'id': 0, 'name': 'DVORANA TABOR', 'bike_stands': 22},
        {'id': 1, 'name': 'EUROPARK - POBREŠKA C.', 'bike_stands': 22},
        {'id': 2, 'name': 'GORKEGA UL. - OŠ FRANCETA PREŠERNA', 'bike_stands': 22},
        {'id': 3, 'name': 'GOSPOSVETSKA C. - III. GIMNAZIJA', 'bike_stands': 22},
        {'id': 4, 'name': 'GOSPOSVETSKA C. - TURNERJEVA UL.', 'bike_stands': 22},
        {'id': 5, 'name': 'GOSPOSVETSKA C. - VRBANSKA C.', 'bike_stands': 22},
        {'id': 6, 'name': 'JHMB – DVOETAŽNI MOST', 'bike_stands': 22},
        {'id': 7, 'name': 'KOROŠKA C. - KOROŠKI VETER', 'bike_stands': 22},
        {'id': 8, 'name': 'LIDL - KOROŠKA C.', 'bike_stands': 22},
        {'id': 9, 'name': 'LIDL - TITOVA C.', 'bike_stands': 22},
        {'id': 10, 'name': 'LJUBLJANSKA UL. - FOCHEVA', 'bike_stands': 22},
        {'id': 11, 'name': 'LJUBLJANSKA UL. - II. GIMNAZIJA', 'bike_stands': 22},
        {'id': 12, 'name': 'MLADINSKA UL. - TRUBARJEVA UL.', 'bike_stands': 22},
        {'id': 13, 'name': 'MLINSKA UL . - AVTOBUSNA POSTAJA', 'bike_stands': 22},
        {'id': 14, 'name': 'NA POLJANAH - HEROJA ŠERCERJA', 'bike_stands': 22},
        {'id': 15, 'name': 'NICEHASH - C PROLETARSKIH BRIGAD', 'bike_stands': 22},
        {'id': 16, 'name': 'NKBM - TRG LEONA ŠTUKLJA', 'bike_stands': 22},
        {'id': 17, 'name': 'PARTIZANSKA C. - CANKARJEVA UL.', 'bike_stands': 22},
        {'id': 18, 'name': 'PARTIZANSKA C. - TIC', 'bike_stands': 22},
        {'id': 19, 'name': 'PARTIZANSKA C. - ŽELEZNIŠKA POSTAJA', 'bike_stands': 22},
        {'id': 20, 'name': 'PETROL – LENT – VODNI STOLP', 'bike_stands': 22},
        {'id': 21, 'name': 'POŠTA - SLOMŠKOV TRG', 'bike_stands': 22},
        {'id': 22, 'name': 'RAZLAGOVA UL. - OBČINA', 'bike_stands': 22},
        {'id': 23, 'name': 'SPAR - TRŽNICA TABOR', 'bike_stands': 22},
        {'id': 24, 'name': 'STROSSMAYERJEVA UL. - TRŽNICA', 'bike_stands': 22},
        {'id': 25, 'name': 'TELEMACH - GLAVNI TRG - STARI PERON', 'bike_stands': 22},
        {'id': 26, 'name': 'ULICA MOŠE PIJADA - UKC', 'bike_stands': 22},
        {'id': 27, 'name': 'UM FGPA - LENT - SODNI STOLP', 'bike_stands': 22},
]


def get_stations():
    for station in stations:
        station['name'] = station['name'].replace('.', '').replace(' ', '_')

    return stations


def get_station_data(station_id):
    for station in stations:
        station['name'] = station['name'].replace('.', '').replace(' ', '_')
        if station['id'] == station_id:
            return station

    return None  # Return None if station_id is not found
