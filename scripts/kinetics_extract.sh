#!/bin/bash

# Download directories vars
root_dl="/home/zhaochenzhi/CRW_TAP/datasets/kinetics700-2020"
root_dl_targz="/home/zhaochenzhi/CRW_TAP/datasets/kinetics700-2020_targz"

# Make root directories
[ ! -d $root_dl ] && mkdir -p $root_dl


# # Extract validation
# curr_dl=$root_dl_targz/val
# curr_extract=$root_dl/val
# [ ! -d $curr_extract ] && mkdir -p $curr_extract
# tar_list=$(ls $curr_dl)
# for f in $tar_list
# do
# 	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
# done

# Extract train
curr_dl=$root_dl_targz/train
curr_extract=$root_dl/train
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extraction complete
echo -e "\nExtractions complete!"