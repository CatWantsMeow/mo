#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import wget
import tarfile


out_dir = 'data/not_mnist'
small_arhive = f'{out_dir}/notMNIST_small.tar.gz'
large_arhive = f'{out_dir}/notMNIST_large.tar.gz'
large_url = 'https://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz'
small_url = 'https://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz'


def download_data():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(small_arhive):
        print(f"Downloading {small_arhive}.")
        wget.download(small_url, small_arhive)
        print()
    else:
        print(f"Skipping {small_arhive} download (already exists)")

    if not os.path.exists(large_arhive):
        print(f"Downloading {large_arhive}.")
        wget.download(large_url, large_arhive)
        print()
    else:
        print(f"Skipping {large_arhive} download (already exists)")


def extract_data():
    print(f"Extracting {small_arhive}")
    with tarfile.open(small_arhive) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, out_dir)

    print(f"Extracting {large_arhive}")
    with tarfile.open(large_arhive) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, out_dir)


if __name__ == '__main__':
    download_data()
    extract_data()
