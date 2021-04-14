import calendar
import datetime


def main():
    print("Su Mo Tu We Th Fr Sa")

    date = datetime.date.today()
    first_day_of_month = date.replace(day=1)
    day_of_week = (first_day_of_month.weekday() + 1) % 7  # 0 -> Sunday, 1 -> Monday, ...
    month_days = calendar.monthrange(date.year, date.month)[1]

    d = 1
    if day_of_week != 0:
        d -= day_of_week
        day_of_week = 0
    for d in range(d, month_days + 1):
        if d >= 1:
            print("%2d " % d, end="")
        else:
            print("   ", end="")
        if day_of_week == 6:
            print()
        day_of_week = (day_of_week + 1) % 7
    if day_of_week != 0:
        print()


if __name__ == '__main__':
    main()
