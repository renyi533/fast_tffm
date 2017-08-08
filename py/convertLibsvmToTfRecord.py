# coding=utf-8

import os
import sys
import io
import tensorflow as tf
import threading
import time
import bisect
import multiprocessing
from struct import *

def convert_to(src, dst):
    with open(src, "r") as f:
        print("Writing", dst)
        writer = tf.python_io.TFRecordWriter(dst)
        for line in f:
            fields = line.strip().split()
            label = float(fields[0])
            fid_arr = []
            val_arr = []
            feature_count = 0
            for i in range(1, len(fields)):
                feature_count += 1
                item_arr = fields[i].split(':')
                fid_arr.append(int(item_arr[0]))
                if len(item_arr)>1:
                    val_arr.append(float(item_arr[1]))
                else:
                    val_arr.append(1.0)

            byte_array = pack('Q', 0x7FFFFFFF7FFFFFFF)
            byte_array = byte_array+pack('I', feature_count) 
            byte_array = byte_array+pack('I', 1)
            for i in range(len(fid_arr)):
                byte_array = byte_array+pack('Q', fid_arr[i])
                byte_array = byte_array+pack('f', val_arr[i])

            byte_array = byte_array + pack('f', label)
            writer.write(byte_array)
        writer.close()

def main(argv):
    start_stamp = time.time()
    convert_to(argv[0], argv[1])
    print("Converting timeuse: %f seconds" % (time.time() - start_stamp))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit("Usage: ConvertToTFRecord.py data output")
    else:
        main(sys.argv[1:])
