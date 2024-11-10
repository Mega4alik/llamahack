# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import codecs
import io
import re
import json
import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
import librosa




def file_put_contents(filename, st):
    file = codecs.open(filename, "w", "utf-8")
    file.write(st)
    file.close()

def file_get_contents(name):
    f = io.open(name, mode="r", encoding="utf-8") #utf-8 | Windows-1252
    return f.read()

def timediff(time1, time2):
    return int( (time2 - time1) * 1000 ) #ms

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def remove_multiple_spaces(text):
    for _ in range(2): text = text.replace("  ", " ")    
    return text

