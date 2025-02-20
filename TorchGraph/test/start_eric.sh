#!/bin/bash
# /opt/nvidia/nsight-compute/2024.3.1/ncu -o test -f --replay-mode kernel --device 0 --nvtx --nvtx-include "Forward/" --nvtx-include "Backward/" --section "SpeedOfLight" --section "LaunchStats" --section "Occupancy" --section "MemoryWorkloadAnalysis" python3 ddp_graph_database_test.py
# /opt/nvidia/nsight-compute/2024.3.1/ncu -o vgg19 -f --replay-mode kernel --device 0 --nvtx --nvtx-include "Forward/" --nvtx-include "Backward/" --section "SpeedOfLight" --section "LaunchStats" --section "Occupancy" --section "MemoryWorkloadAnalysis" python3 ddp_graph_database_test.py --model vgg19 --type CV --batchsize 64
/opt/nvidia/nsight-compute/2024.3.1/ncu -o resnet152 -f --replay-mode kernel --device 0 --nvtx --nvtx-include "Forward/" --nvtx-include "Backward/" --section "SpeedOfLight" --section "LaunchStats" --section "Occupancy" --section "MemoryWorkloadAnalysis" python3 ddp_graph_database_test.py --model resnet152 --type CV --batchsize 64
# /opt/nvidia/nsight-compute/2024.3.1/ncu -o gpt2 -f --replay-mode kernel --device 0 --nvtx --nvtx-include "Forward/" --nvtx-include "Backward/" --section "SpeedOfLight" --section "LaunchStats" --section "Occupancy" --section "MemoryWorkloadAnalysis" python3 ddp_graph_database_test.py --model gpt2 --type NLP --batchsize 16

# Convert the output file to CSV format
# /opt/nvidia/nsight-compute/2024.3.1/ncu -i "vgg19.ncu-rep" --page details --csv --log-file "vgg19.csv"
/opt/nvidia/nsight-compute/2024.3.1/ncu -i "resnet152.ncu-rep" --page details --csv --log-file "resnet152.csv"
# /opt/nvidia/nsight-compute/2024.3.1/ncu -i "gpt2.ncu-rep" --page details --csv --log-file "gpt2.csv"
