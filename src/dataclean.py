import json

quarters = ['01-01', '04-01', '07-01', '10-01']

def frmt_quarterise(filename):
    data_str = filename.read()
    data_d = [x.split(',') for x in data_str.split('\n')[1:] if x]
    data_totals = []
    prev_q = 0
    q_index = -1
    for x in range(len(data_d)):
        y, m, d = [int(y) for y in data_d[x][0].split('-')]
        q = (m-1)//3 + 1
        if q == prev_q:
            if data_d[x][1]:
                data_totals[q_index][1].append(float(data_d[x][1]))
                data_totals[q_index][2] += 1
        else:
            q_index += 1
            if data_d[x][1]:
                data_totals.append([f"{y}-{quarters[q-1]}", [float(data_d[x][1])], 1])
            else:
                data_totals.append([f"{y}-{quarters[q-1]}", [], 0])
        prev_q = q
    data_list = [[x[0], sum(x[1])/x[2]] for x in data_totals]
    data = {x[0]:x[1] for x in data_list}
    return data

# def deseasonalise(data):
#     dates = data.keys()
#     values = data.values()
#     avg = sum(values)/len(values)
#     quarter_totals = [[[], 0]*4]
#     for i in range(len(dates)):
#         date = dates[i]
#         value = values[i]
#         quarter_i = quarters.index(date[5:])
#         quarter_totals[quarter_i][0].append(value)
#         quarter_totals[quarter_i][1] += 1
#     quarter_avgs = [sum(quarter_totals[x][0])/quarter_totals[x][1] for x in range(4)]
#     proc_data = []
#     for i in range(len(dates)):
#         date = dates[i]
#         value = values[i]
#         quarter_i = quarters.index(date[5:])
#         value += avg - quarter_avgs[quarter_i]
#         proc_data.append(value)
#     return dict(zip(dates, proc_data))


with open(f"../data/{input("gdp filename? ")}") as gdp_file:
    gdp = frmt_quarterise(gdp_file)

with open(f"../data/{input("cpi filename? ")}") as cpi_file:
    cpi = frmt_quarterise(cpi_file)

with open(f"../data/{input("10y 3m yield spread filename? ")}") as yspread_file:
    yspread = frmt_quarterise(yspread_file)

dates = sorted(list(set(gdp.keys()) & set(cpi.keys()) & set(yspread.keys())))

data = {date:[gdp[date], cpi[date], yspread[date]] for date in dates}

with open(f"../data/{input("output filename (.json)? ")}.json", 'w') as json_file:
    json.dump(data, json_file)