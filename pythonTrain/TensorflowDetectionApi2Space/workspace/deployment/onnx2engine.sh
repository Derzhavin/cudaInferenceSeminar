trtexec --onnx=$1 --fp16 --useCudaGraph --noDataTransfers --workspace=4096  --buildOnly --saveEngine=$2
