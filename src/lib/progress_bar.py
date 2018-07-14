# -*- coding: utf-8 -*-
"""
Created on 2018/7/13

@author: Free_QC
"""
import sys


class ProgressBar:
    def __init__(self, count=0, total=0, width=50):
        self.count = count
        self.total = total
        self.width = width

    def move(self):
        self.count += 1

    def log(self, s):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = self.width * self.count // self.total
        sys.stdout.write('{0},{1:3}/{2:3}: '.format(s, self.count, self.total))
        sys.stdout.write('#' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()
