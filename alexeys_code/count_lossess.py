#!/usr/bin/env python
import numpy as np

def count_losses(timelost : np.ndarray, tmax: float = 1e-3) -> None:
    count = 0
    for t in timelost:
        count += (t < tmax)
    print(f'Lost particle {count=}')
    print(f'This is {count/len(timelost)} of all partickes')

print('\n=== CPU: ===\n')
count_losses(np.loadtxt('output/timelost.txt'))

print('\n=== GPU: ===\n')
count_losses(np.loadtxt('output_gpu/timelost.txt'))
