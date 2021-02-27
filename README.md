# Another NumPy Circular Buffer

[![Build Status](https://travis-ci.com/EmDash00/ANCB.svg?branch=master)](https://travis-ci.com/EmDash00/ANCB)

Another NumPy Circular Buffer (or ANCB for short) is an attempt to make a circular buffer work with NumPy ufuncs for
real-time data processing. One can think of a NumpyCircularbuffer in ANCB as being a fixed length deque with random access
functionality (unlike the deque). For users more familar with NumPy, one can think of this buffer as a way of automatically
rolling the array into the right order.

ANCB was developed by Drason "Emmy" Chow during their time as an undergraduate researcher at IU: Bloomington for use in 
making [Savitzky-Golay filters](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter), which take an array of positions in chronological or reverse-chronological order and produce
estimates of velocity, acceleration, and possibly higher order derivatives if desired.

Looking for the documentation? You can find it here:  
https://ancb-docs.readthedocs.io/en/latest/
