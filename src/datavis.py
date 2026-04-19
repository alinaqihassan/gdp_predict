import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

with open('../data/quarterly_data.json', 'r') as file:
    data = json.load(file)

dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in data.keys()]

gdp = [x[0] for x in data.values()]
cpi = [x[1] for x in data.values()]
yspread = [x[2] for x in data.values()]

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator(base=4))

ax1.plot(dates, gdp)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator(base=4))

ax2.plot(dates, cpi)
ax2.plot(dates, yspread)

fig.autofmt_xdate()

plt.show()