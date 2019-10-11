import statistics


ages = [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70]
# 3.7(a)
ages_min = 13
ages_max = 70
ages_ans1 = (35 - ages_min) / (ages_max - ages_min) * (1 - 0) + 0
print('%.2f' % ages_ans1)

# 3.7(b)
ages_mean = statistics.mean(ages)
ages_standard_deviation = statistics.stdev(ages)
print('%.2f' % ages_mean, '%.2f' % ages_standard_deviation)
ages_ans2 = (35 - ages_mean) / ages_standard_deviation
print('%.2f' %ages_ans2)

# 3.7(c)
ages_ans3 = 35 / 100
print('%.2f' %ages_ans3)

# 3.7(d)
print('I prefer decimal scaling for given data. Since this kind of normalization can keep the original data distribution, and the actual meaning is more straight forward to users.')


