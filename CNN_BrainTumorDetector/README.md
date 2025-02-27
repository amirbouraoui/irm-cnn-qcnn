# Setup 
conda create -n BTD python=3.11
conda activate BTD 
python -m ipykernel install --user --name=BTD --display-name="Python (BTD)"
pip install -r requirements.txt
jupyter notebook