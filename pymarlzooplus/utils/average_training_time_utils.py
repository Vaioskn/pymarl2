def parse_time(time_str):
    """ Parses a time string formatted as 'Xd : Yh' and returns the total hours. """
    parts = time_str.split(':')
    days = int(parts[0].strip().replace('d', ''))
    hours = int(parts[1].strip().replace('h', ''))
    total_hours = days * 24 + hours
    return total_hours


def average_training_time(times):
    """ Computes the average training time from a list of time strings. """
    total_hours = sum(parse_time(time) for time in times)
    avg_hours = total_hours / len(times)
    avg_days = avg_hours // 24
    avg_remain_hours = int(avg_hours % 24)
    return f"{int(avg_days)}d : {avg_remain_hours}h"


training_times = ["1d : 16h", "2d : 10h"]
average_time = average_training_time(training_times)
print("Average Training Time:", average_time)
