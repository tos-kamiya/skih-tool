# Skih-tool

A tool for predicting lines where end-of-sentence comments appear using DNN.

## Install dependencies

1. Install tensorflow for CPU or GPU, depending on your environment.
2. Install other dependencies listed in `requirements.txt`.

E.g., 

```sh
python3 -m pip install tensorflow
python3 -m pip install -r requirements.txt
```

## Run

Run the script `apply_model.py` with a programming language, a threshold (the larger the threshold, the less end-of-sentence comments are output), and a source file.

```
python3 apply_model.py -l <language> -p <threshold> <sourcefile>
```

The tensorflow log will be output to the standard error output, so redirect if you do not want to see the log.

E.g.,

```sh
python3 apply_model.py -l python -p 0.7 main.py 2> /dev/null
```

```sh
python3 apply_model.py -l java -p 0.7 Main.java 2> /dev/null
```

The output will be for each line of the target source file given as an argument, with a "*" at the beginning of the line where the end-of-sentence comment is predicted to be.

```sh
$ python3 apply_model.py -l java -p 0.5 samples/MonthCalendar.java 2> /dev/null
- samples/MonthCalendar.java
  import java.time.LocalDate;
  
  public class MonthCalendar {
      public static void main(String[] args) {
          System.out.println("Su Mo Tu We Th Fr Sa");
  
          LocalDate date = LocalDate.now();
          LocalDate firstDayOfMonth = LocalDate.of(date.getYear(), date.getMonthValue(), 1);
          int dayOfWeek = firstDayOfMonth.getDayOfWeek().getValue();
          int monthDays = date.lengthOfMonth();
  
*         int d = 1;
          if (dayOfWeek != 1) {
              d -= dayOfWeek - 1;
*             dayOfWeek = 1;
          }
          for (; d <= monthDays; d++) {
              if (d >= 1) {
                  System.out.printf("%2d ", d);
              } else {
                  System.out.printf("   ");
              }
              if (dayOfWeek == 7) {
                  System.out.println();
              }
*             dayOfWeek = dayOfWeek % 7 + 1;
          }
          if (dayOfWeek != 1) {
              System.out.println();
          }
      }
  }
```

Another example:

```sh
python3 /apply_model.py -p 0.7 -l python samples/month_calendar.py 2> /dev/null
- samples/month_calendar.py
  import calendar
  import datetime
  
  
  def main():
      print("Su Mo Tu We Th Fr Sa")
  
      date = datetime.date.today()
*     first_day_of_month = date.replace(day=1)
*     day_of_week = (first_day_of_month.weekday() + 1) % 7  # 0 -> Sunday, 1 -> Monday, ...
      month_days = calendar.monthrange(date.year, date.month)[1]
  
      d = 1
      if day_of_week != 0:
          d -= day_of_week
*         day_of_week = 0
      for d in range(d, month_days + 1):
          if d >= 1:
              print("%2d " % d, end="")
          else:
              print("   ", end="")
          if day_of_week == 6:
*             print()
*         day_of_week = (day_of_week + 1) % 7
      if day_of_week != 0:
*         print()
  
  
* if __name__ == '__main__':
      main()
```