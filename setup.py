#!/usr/bin/python

from distutils.core import setup
from glob import glob
from itertools import chain
from os.path import isdir
from shutil import copy, rmtree
from subprocess import call
from sys import path

import py2exe # pylint: disable=F0401,W0611

print "Starting..."

PROGRAM_NAME = 'Signs'

BUILD_NAME = 'build'
MSVC_SEARCH = r'C:\Program Files*'
MSVC_DIR = 'Microsoft.VC90.CRT'

options = dict(py2exe = dict(
    dist_dir = PROGRAM_NAME, unbuffered = True, ascii = True,
    bundle_files = 1, optimize = 2, compressed = True,
    excludes = ['difflib', 'doctest', 'hashlib', 'locale', 'optparse', 'pickle', 'calendar', 'serialposix', 'serialjava', 'serialcli', 'tcl']
))

msvcPaths = glob(r'%s\*\%s' % (MSVC_SEARCH, MSVC_DIR)) + glob(r'%s\*\*\%s' % (MSVC_SEARCH, MSVC_DIR))

(msvcPath, files) = ((path, files) for (path, files) in ((path, glob(path + r'\*.*')) for path in msvcPaths) if len(files) == 4).next()

path.append(msvcPath)

print "Cleaning..."

if isdir(BUILD_NAME):
    rmtree(BUILD_NAME)
if isdir(PROGRAM_NAME):
    rmtree(PROGRAM_NAME)

print "Building..."

setup(version = '1.0',
      description = 'Wall of Signs for "House, where the World Sounds..." Russian LARP',
      author = 'Vasily Zakharov',
      author_email = 'vmzakhar@gmail.com',
      console = (PROGRAM_NAME + '.py',),
      data_files = ((MSVC_DIR, files),),
      options = options)

print "Packaging..."

for fileName in chain(glob('*.py'), glob('*.cmd'), glob('*.txt')):
    copy(fileName, PROGRAM_NAME)

call(('zip', '-r', '-9', PROGRAM_NAME + '.zip', PROGRAM_NAME), shell = True)

print "\nCleaning..."

if isdir(BUILD_NAME):
    rmtree(BUILD_NAME)
if isdir(PROGRAM_NAME):
    rmtree(PROGRAM_NAME)

print "Done"
