The files here contain the aggregate statistics for the MFCC data given by the
Kaggle competition.  Each file consists of a matrix.  Each row of the matrix
corresponds to a training record (i.e. an audio file).  The columns represent
aggregation statistics for each mel frequency cepstral coefficient (MFCC).
In the MFCC data file given by Kaggle, there is a 17xn matrix with the rows
being the MFCCs and the columns being time frames.  In my transformation,
I calculate aggregation statistics on each MFCC (each row) of the file.
For example, I create average, median, mean absolute deviation, and standard deviation
for each MFCC in the Kaggle MFCC data.  I then concatenate the aggregation statistics
for all of the MFCCs for an audio file and make that a line in the new file
(each line in the new file is a feature vector).

An example of what a file 'schema' looks like for a file (i.e. file001.wav)
Note mad is mean absolute deviation; stdev is standard deviation
average_MFCC1 median_MFCC1 mad_MFCC1 stdev_MFCC1 ... continues 2-16 ... average_MFCC17 median_MFCC17 mad_MFCC17 stdev_MFCC17
