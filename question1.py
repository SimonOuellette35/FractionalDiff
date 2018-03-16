import numpy as np

NUM_RWS = 1000      # 1000 random walks
RW_LENGTH = 10000   # 10,000 data points per random walk

#a) build several random walks
print "Building random walks..."
random_walks = []
for i in range(NUM_RWS):
    rw_i = []
    last = 0.0
    for j in range(RW_LENGTH):
        new_val = last + np.random.normal(0.0, 1.0, 1)[0]
        rw_i.append(new_val)
        last = new_val

    print str(((i + 1) / float(NUM_RWS)) * 100.0) + "%% complete"
    random_walks.append(rw_i)

print "Running I(1) correlations..."
#b) correlate them between each other, as is, in their original I(1) form. Estimate the amount of spurious correlation.
rho_list = []

for i in range(1, NUM_RWS):
    rho = np.corrcoef(random_walks[i-1], random_walks[i])[1][0]
    rho_list.append(rho)
    print str((i / float(NUM_RWS)) * 100.0) + "%% complete"

print "Done. Rho array: ", rho_list

#c) differentiate (with d = 1) these time series, and produce their stationary I(0) equivalents.
full_diffs = []

print "Differentiating with d = 1"
for i in range(NUM_RWS):
    diff = []
    for j in range(1, RW_LENGTH):
        diff.append(random_walks[i][j] - random_walks[i][j-1])

    full_diffs.append(diff)
    print str(((i+1) / float(NUM_RWS)) * 100.0) + "%% complete"

#d) re-run all correlations as in b), but on the differentiated time series. Estimate results.
rho_list2 = []

print "Done. Running I(0) correlations..."
for i in range(1, NUM_RWS):
    rho2 = np.corrcoef(full_diffs[i-1], full_diffs[i])[1][0]
    rho_list2.append(rho2)
    print str((i / float(NUM_RWS)) * 100.0) + "%% complete"

print "Done. Rho array #2: ", rho_list2

import pylab as pl

# from: http://www.mirzatrokic.ca/FILES/codes/fracdiff.py
# small modification: wrapped 2**np.ceil(...) around int()
def fracdiff( x,d ):
    T=len(x)
    np2=int(2**np.ceil(np.log2(2*T-1)))
    k=np.arange(1,T)
    b=(1,) + tuple(np.cumprod((k-d-1)/k))
    z = (0,)*(np2-T)
    z1=b + z
    z2=tuple(x) + z
    dx = pl.ifft(pl.fft(z1)*pl.fft(z2))
    return pl.real(dx[0:T])

def fd(d):
    print "Differencing with d = %s..." % d
    frac_diffs = []
    for i in range(NUM_RWS):
        frac = fracdiff(np.array(random_walks[i]), d)
        frac_diffs.append(frac)
        print str(((i + 1) / float(NUM_RWS)) * 100.0) + "%% complete"

    rho_list3 = []
    print "Done. Producing correlations..."
    for i in range(1, NUM_RWS):
        rho3 = abs(np.corrcoef(frac_diffs[i - 1], frac_diffs[i])[1][0])
        rho_list3.append(rho3)
        print str((i / float(NUM_RWS)) * 100.0) + "%% complete"

    print "Done. Rho array: ", rho_list3

    return np.mean(rho_list3)

mean_rho = []
for d in range(0, 11):
    avg = fd(float(d) / 10.0)
    mean_rho.append(avg)

print "mean rhos = ", mean_rho
