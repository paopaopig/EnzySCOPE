python .\src\embed_molecules.py --molecule-csv .\data\train\train_molecules.csv --output-pkl .\data\train\train_molecule_embeddings.pkl
python .\src\embed_enzymes.py --enzyme-csv .\data\train\train_enzymes.csv --output-pt .\data\train\train_enzyme_embeddings.pt
python .\src\build_graphs.py --enzyme-csv .\data\train\train_enzymes.csv --structure-dir .\raw_structures --output-dir .\data\train\train_protein_graphs
python .\src\train.py --config .\configs\train_config.json --epochs 100 --seed 1234
python .\src\predict.py
