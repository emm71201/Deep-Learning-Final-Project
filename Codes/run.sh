echo $"\nDownloading data and metadata"
python3 download_data.py
echo $"\nRun GMLP, ResNet50 with Metadata, CNN with Attention"
echo $"\n1. Gated MLP"
python3 main_code.py gmlp 32 0.001
echo $"\n2. RESNET50 with Metadata"
python3 main_code.py resnet50_model 64 0.001
echo $"\n2. CNN with Attention"
python3 main_code.py attentioncnn_model 32 0.001

