import statistics
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples

ages = [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61]
fats = [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]

#2.4(1）
age_mean = statistics.mean(ages)
age_median = statistics.median(ages)
age_deviation = statistics.pstdev(ages)
print(age_mean, age_median, age_deviation)

#2.4(2）
plt.boxplot(ages, patch_artist=True, labels=['ages'])
plt.show()
plt.boxplot(fats, patch_artist=True, labels=['fats%'])
plt.show()

#2.4(3）
plt.scatter(ages, fats)
plt.xlabel("ages")
plt.ylabel("fats%")
plt.show()

ages_array = np.asarray(ages)
fats_array = np.asarray(fats)
pp_ages = sm.ProbPlot(ages_array)
pp_fats = sm.ProbPlot(fats_array)
qqplot_2samples(pp_fats, pp_ages, xlabel='ages', ylabel='fats%', line='r')
plt.show()