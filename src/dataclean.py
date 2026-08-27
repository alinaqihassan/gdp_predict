# dataclean.py
# reads the three data files, converts them to dictionaries, combines them into a single dictionary,
# and saves them to a json file

import json

quarters = ['01-01', '04-01', '07-01', '10-01']

# reads from the files and creates a dictionary with the data values for each quarter,
# averaging readings at a higher frequency
def frmt_quarterise(file):
    data_str = file.read()
    data_d = [x.split(',') for x in data_str.split('\n')[1:] if x]
    data_totals = []
    prev_quarter = 0
    quarter_i = -1
    for x in range(len(data_d)):
        y, m, d = [int(y) for y in data_d[x][0].split('-')]
        q = (m-1)//3 + 1
        if q == prev_quarter:
            if data_d[x][1]:
                data_totals[quarter_i][1].append(float(data_d[x][1]))
                data_totals[quarter_i][2] += 1
        else:
            quarter_i += 1
            if data_d[x][1]:
                data_totals.append([f"{y}-{quarters[q-1]}", [float(data_d[x][1])], 1])
            else:
                data_totals.append([f"{y}-{quarters[q-1]}", [], 0])
        prev_quarter = q
    data_list = [[x[0], sum(x[1])/x[2]] for x in data_totals]
    data = {x[0]:x[1] for x in data_list}
    return data


with open(f"../data/{input("gdp filename? ")}") as gdp_file:
    gdp = frmt_quarterise(gdp_file)

with open(f"../data/{input("cpi filename? ")}") as cpi_file:
    cpi = frmt_quarterise(cpi_file)

with open(f"../data/{input("10y 3m yield spread filename? ")}") as yspread_file:
    yspread = frmt_quarterise(yspread_file)

dates = sorted(list(set(gdp.keys()) & set(cpi.keys()) & set(yspread.keys()))) # sorted list of dates that only appear in all three datasets

data = {date:[gdp[date], cpi[date], yspread[date]] for date in dates}

with open(f"../data/{input("output filename (.json)? ")}.json", 'w') as json_file:
    json.dump(data, json_file)