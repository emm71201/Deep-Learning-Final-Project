echo $"\nDownloading data and metadata"
python3 download_data.py
echo $"\n Gated MLP"
python3 main_code.py gmlp

