echo $"\nDownloading data and metadata"
python3 download_data.py
echo $"\nRun GMLP, ResNet50 with Metadata, CNN with Attention"
echo $"\n1. Gated MLP"
python3 main_code.py gmlp
echo $"\n2. RESNET50 with Metadata"
python3 main_code resnet50_model
echo $"\n2. CNN with Attention"
python3 main_code attentioncnn_model

