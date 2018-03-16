import ornstein_uhlenbeck as ou
import numpy as np
import statsmodels.tsa.stattools as st
from scipy import stats

TS_LENGTH = 50000
STARTING_LEVEL = 5.0
MEAN_REVERT = 0.3
LONG_TERM_MEAN = 5.5
VALIDATION_AMOUNT = 5000
TREND = 0.005

# a) build an Ornstein-Uhlenback process with trend, to make it non-stationary.
# dx(t) = theta((mu + trend) - x(t)) dt + sigma * dW(t), where theta > 0, mu and sigma > 0, trend != 0
param = ou.ModelParameters(TS_LENGTH, STARTING_LEVEL, MEAN_REVERT, LONG_TERM_MEAN, TREND)
ou_time_series = ou.ornstein_uhlenbeck_levels(param)

# first test that the time series is non-stationary, with an ADF test -- later use the same ADF
# test to confirm that the d choice is optimal.
result = st.adfuller(ou_time_series)
print "OU time series' ADF test result: ", result

# b) difference it with d = 1
ou_diffs = []
for i in range(1, TS_LENGTH):
    ou_diffs.append(ou_time_series[i] - ou_time_series[i - 1])

# c) build regression model using I(0) as feature, estimate prediction power out of sample
# model: dx(t) = alpha + beta * dx(t-1)
y = ou_diffs[1:]
y_lagged = ou_diffs[:-1]

def regress_and_validate(y, y_lagged):
    training_y = []
    training_x = []

    validation_y = []
    validation_x = []

    for i in range(len(y) - VALIDATION_AMOUNT):
        training_y.append(y[i])
        training_x.append(y_lagged[i])

    for i in range(len(y) - VALIDATION_AMOUNT, len(y)):
        validation_y.append(y[i])
        validation_x.append(y_lagged[i])

    result = stats.linregress(training_x, training_y)
    print "Regression result:", result

    # estimate out of sample fit
    intercept = result[1]
    slope = result[0]

    avg_y = sum(validation_y) / float(len(validation_y))

    sse = 0.0
    sst = 0.0
    for i in range(len(validation_y)):
        estimate = (slope * validation_x[i]) + intercept
        error = estimate - validation_y[i]
        sse += error ** 2.0
        sst += (validation_y[i] - avg_y) ** 2.0

    MSE = sse / float(len(validation_y))
    R2 = 1.0 - (sse / sst)
    variance = sst / float(len(validation_y))

    print "Out of sample R2 = %s (MSE = %s, variance of y = %s)" % (R2, MSE, variance)

regress_and_validate(y, y_lagged)

import pylab as pl

# from: http://www.mirzatrokic.ca/FILES/codes/fracdiff.py
# small modification: wrapped 2**np.ceil(...) around int()
def fracdiff(x,d):
    T=len(x)
    np2=int(2**np.ceil(np.log2(2*T-1)))
    k=np.arange(1,T)
    b=(1,) + tuple(np.cumprod((k-d-1)/k))
    z = (0,)*(np2-T)
    z1=b + z
    z2=tuple(x) + z
    dx = pl.ifft(pl.fft(z1)*pl.fft(z2))
    return pl.real(dx[0:T])

# d) fractionally difference it (optimal d = ?)
# NOTE: here I just manually ran all values of d described in the blog post... I could have simply
# created a for loop, just like in question1.py, but I didn't...

d = 0.9 # choose your d value...

print "Differencing with d = %s..." % d
frac_diffs = fracdiff(ou_time_series, d)

# check if we became stationary by running an ADF test
result2 = st.adfuller(frac_diffs)

print "Fractionally differenced time series' ADF result: ", result2

# build regression model using I(fractional) as feature, estimate prediction power out of sample
frac_lagged = frac_diffs[1:-1]

regress_and_validate(y, frac_lagged)
