# Skih-tool

DNNを使って文末コメントが出現する行を予測するツールです。

## 依存関係のインストール

1. 環境に応じてCPU用またはGPU用のtensorflowをインストールする。
2. `requirements.txt`に記載されているその他の依存関係をインストールする。

例えば、

```
python3 -m pip install tensorflow
python3 -m pip install -r requirements.txt
```

## 実行

スクリプト`apply_model.py`を、プログラミング言語、しきい値（しきい値が大きいほど文末のコメントが出力されなくなる）、ソースファイルを指定して実行します。

```
python3 apply_model.py -l <language> -p <threshold> <sourcefile>
```

tensorflowのログは標準エラー出力に出力されますので、ログを見たくない場合はリダイレクトしてください。

(例)

```sh
python3 apply_model.py -l python -p 0.7 main.py 2> /dev/null
```

```sh
python3 apply_model.py -l java -p 0.7 Main.java 2> /dev/null
```

出力は、引数として与えた対象ソースファイルの各行に対して、文末コメントがあると予測した行の行頭に「*」をつけたものとなります。

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

別の例

```sh
python3 /apply_model.py -p 0.7 -l python samples/month_calendar.py 2> /dev/null
- samples/month_calendar.py
  import calendar
  import datetime
  
  
  def main():
      print("Su Mo Tu We Th Fr Sa")
  
*     date = datetime.date.today()
*     first_day_of_month = date.replace(day=1)
*     day_of_week = (first_day_of_month.weekday() + 1) % 7  # 0 -> Sunday, 1 -> Monday, ...
*     month_days = calendar.monthrange(date.year, date.month)[1]
  
*     d = 1
*     if day_of_week != 0:
          d -= day_of_week
*         day_of_week = 0
      for d in range(d, month_days + 1):
          if d >= 1:
              print("%2d " % d, end="")
          else:
              print("   ", end="")
*         if day_of_week == 6:
*             print()
*         day_of_week = (day_of_week + 1) % 7
*     if day_of_week != 0:
*         print()
  
  
* if __name__ == '__main__':
      main()
```

## 設計、アルゴリズムなど

次を参照してください。

神谷 年洋, "コードレビュー向けコメント行位置予測ツールの試作", 信学技報, Vol. 120, No. 193, SS2020-12, DC2020-29, pp. 43-48 (2020-10-19)
https://www.ieice.org/publications/ken/summary.php?contribution_id=110154

ただし、このページで配布されている予測モデルは発表後に作成されたものです。

* java.hdf5, java.piclkeは発表の後にチューニングされて性能が向上しています。
* python.hdf5, python.pickleは発表後新たにPythonのソースコード向けに作成されたものです。