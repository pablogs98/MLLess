# Authors: Marc Sanchez Artigas <marc.sanchez@urv.cat>
# License: BSD 3 clause
from scheduler import WorkerScheduler, ReclaimAction
import numpy as np
import pandas as pd
import sys

def ewma_(x):
    df = pd.Series(x)
    fwd = pd.Series.ewm(df,span=20).mean() # take EWMA in fwd direction
    bwd = pd.Series.ewm(df[::-1],span=20).mean() # take EWMA in bwd direction
    return  np.mean(np.vstack(( fwd, bwd[::-1] )), axis=0)

def prepare(filename):
    df = pd.read_csv(filename)
    diff = np.diff(df['time'].values)
    df.drop(df.head(1).index, inplace=True)
    df['time'] = diff
    return df

if __name__ == "__main__":
    assert len(sys.argv) == 3, 'Expected two arguments: dataset; threshold'
    filename, threshold = sys.argv[1], float(sys.argv[2])

    # load dataset
    data = prepare(filename)

    # convergence
    loss = ewma_(data['loss'].values)
    convergence_step = np.argmax(loss < threshold)
    convergence_time = np.sum(data['time'].iloc[:convergence_step])
    print('Convergence step {:3d}, time {:2.4f} s'. \
                format(convergence_step, convergence_time))

    ini_workers = data['workers'].iloc[0]
    sched = WorkerScheduler(ini_workers)

    t = interval = 25   # scheduling interval
    context = { 'threshold': threshold }
    for step in range(convergence_step):
        sched.update(step,                     # step number
                    data['loss'].iloc[step],   # loss value
                    data['time'].iloc[step])   # step duration

        if not sched.is_in_stationary_regime(verbose=True):
            continue

        if t >= interval:
            t = 0
            context['current_step'] = step
            action = sched.schedule(context, verbose=True)
            if type(action) == ReclaimAction:
                sched.scale_down(verbose=True)
        else:
            t += data['time'].iloc[step]
