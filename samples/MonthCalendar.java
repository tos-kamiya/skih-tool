import java.time.LocalDate;

public class MonthCalendar {
    public static void main(String[] args) {
        System.out.println("Su Mo Tu We Th Fr Sa");

        LocalDate date = LocalDate.now();
        LocalDate firstDayOfMonth = LocalDate.of(date.getYear(), date.getMonthValue(), 1);
        int dayOfWeek = firstDayOfMonth.getDayOfWeek().getValue();
        int monthDays = date.lengthOfMonth();

        int d = 1;
        if (dayOfWeek != 1) {
            d -= dayOfWeek - 1;
            dayOfWeek = 1;
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
            dayOfWeek = dayOfWeek % 7 + 1;
        }
        if (dayOfWeek != 1) {
            System.out.println();
        }
    }
}

