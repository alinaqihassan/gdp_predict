import json

quarters = ['01-01', '04-01', '07-01', '10-01']

with open('../data/CPALTT01USQ657N.csv') as cpi_file:
    cpi_str = cpi_file.read()
    cpi_list = [x.split(',') for x in cpi_str.split('\n')[1:] if x]
    cpi = {x[0]:float(x[1]) for x in cpi_list}

with open('../data/GDP.csv') as gdp_file:
    gdp_str = gdp_file.read()
    gdp_list = [x.split(',') for x in gdp_str.split('\n')[1:] if x]
    gdp = {x[0]:float(x[1]) for x in gdp_list}

with open('../data/T10Y3M.csv') as yspread_file:
    yspread_str = yspread_file.read()
    yspread_d = [x.split(',') for x in yspread_str.split('\n')[1:] if x]
    yspread_totals = []
    prev_q = 0
    q_index = -1
    for x in range(len(yspread_d)):
        y, m, d = [int(y) for y in yspread_d[x][0].split('-')]
        q = (m-1)//3 + 1
        if q == prev_q:
            if yspread_d[x][1]:
                yspread_totals[q_index][1].append(float(yspread_d[x][1]))
                yspread_totals[q_index][2] += 1
        else:
            q_index += 1
            if yspread_d[x][1]:
                yspread_totals.append([f"{y}-{quarters[q-1]}", [float(yspread_d[x][1])], 1])
            else:
                yspread_totals.append([f"{y}-{quarters[q-1]}", [], 0])
        prev_q = q
    yspread_list = [[x[0], sum(x[1])/x[2]] for x in yspread_totals]
    yspread = {x[0]:x[1] for x in yspread_list}

dates = sorted(list(set(gdp.keys()) & set(cpi.keys()) & set(yspread.keys())))

data = {date:[gdp[date], cpi[date], yspread[date]] for date in dates}

with open('../data/quarterly_data.json', 'w') as json_file:
    json.dump(data, json_file)