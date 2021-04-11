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
