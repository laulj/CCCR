#!/usr/bin/env python3
# This is the post-processing script develop for an ACCR system
# to verify the validity of CCC by its check digit.
import numpy as np

def ccc_validate(cd_str):
    cd = 11
    idx = 0
    ccc_digit = []
    res = 0
    ccc_sum = 0
    # convert CCC into capital letters
    cd_str = cd_str.upper()
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    # S1 Convert alphabets into digits
    for ccc in cd_str:
        if ccc.isalpha() == True and idx <= 4:
            temp = decrypt(ccc)
            ccc_digit.append(int(temp))
        elif ccc.isdigit() == True:
            ccc_digit.append(int(ccc))
        idx += 1
        
    # S2 Sum all the digits
    if len(ccc_digit) == 11:
        cd = int(ccc_digit[10])
        ccc_sum = np.dot(ccc_digit[0:10], p2)
    else:
        return "FAILED"
#    print(ccc_sum)    
    # S3 Taking the modulo of the sum
    res = sum_digit(ccc_sum)
#    print (res)
    # Check if res == check digit
    if res == cd:
        return "PASSED"
    else:
        return "FAILED"

def decrypt(ccc):
    dict_ccc = {
        "A": 10, "B": 12, "C": 13, "D": 14,
        "E": 15, "F": 16, "G": 17, "H": 18,
        "I": 19, "J": 20, "K": 21, "L": 23,
        "M": 24, "N": 25, "O": 26, "P": 27,
        "Q": 28, "R": 29, "S": 30, "T": 31,
        "U": 32, "V": 34, "W": 35, "X": 36,
        "Y": 37, "Z": 38
    }
    if dict_ccc.get(ccc) is not None:
        return dict_ccc.get(ccc)
    else:
        return "ERROR"
        
def sum_digit(num):
    sd = 0
    if num % 11 == 10:
        sd = 0
    elif num % 11 > 10:
        i = num
        while i % 10 > 0:
            rem = int(i % 10)
            i = int(i/10)
            sd = sd + rem
    else:
        sd = num % 11
    return sd