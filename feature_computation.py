import json
import pandas as pd
from itertools import product
import os
import logging
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)


amino_symbol = []
amino_atoms = []
amino_atom_composition = {}
bonds_data = {}
output_file = ""
physio_chemical_properties = {}


def load_protein_info(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            logging.info("Protein information loaded successfully.")

            return {
                "amino_symbol": data.get("amino_Symbols", []),
                "atomic_structure": data.get("protein_Atomic_Structure", {}),
                "amino_atom_composition": data.get("amino_Atom_Composition", {}),
                "amino_atoms": data.get("amino_Atoms", []),
                "bonds_data": data.get("bonds_Data", {}),
                "physio_chemical_properties" : data.get("physio_chemical_properties" , {})
            }
    
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in {file_path}")
    except KeyError as e:
        logging.error(f"Error: Missing key in JSON - {e}")
    except Exception as e:
        logging.critical(f"Unexpected error: {e}")
    
    return {"amino_symbol": [], "amino_atoms": [], "amino_atom_composition": {}, "bonds_data": {}, "physio_chemical_properties" : {}}



def _get_peptide_name(peptide_size):
    peptide_names = {
        1: "Mono", 2: "Di", 3: "Tri", 4: "Tetra", 5: "Penta", 
        6: "Hexa", 7: "Hepta", 8: "Octa", 9: "Nona", 10: "Deca"
    }
    return peptide_names.get(peptide_size, f"{peptide_size}-Peptide")


def PolyP_Comp(sequence, peptide_size, order):
    logging.debug(f"Computing PolyP_Comp for sequence {sequence[:10]}... size={peptide_size}, order={order}")

    peptide_name = _get_peptide_name(peptide_size)
    length = len(sequence)
    poly_dict = {f"{peptide_name}{order}_{''.join(p)}": 0.0 for p in product(amino_symbol, repeat=peptide_size)}
    
    if length < peptide_size:
        return poly_dict  
    
    for i in range(length - (peptide_size - 1) * order):
        peptide = ''.join(sequence[i + j * order] for j in range(peptide_size))
        key = f"{peptide_name}{order}_{peptide}"
        if key in poly_dict:
            poly_dict[key] += 1
    
    total_peptides = sum(poly_dict.values())
    if total_peptides > 0:
        for key in poly_dict:
            poly_dict[key] = (poly_dict[key] / total_peptides) * 100
    
    return poly_dict

def ATC(sequence):
    amino_count = {a: 0 for a in amino_symbol}
    atoms = {"ATC_" + a: 0 for a in amino_atoms}

    for a in sequence:
        if a in amino_count:
            amino_count[a] += 1
        else :
            logging.warning(f"Unknown amino acid found: {a}")
            continue
    
    for a in amino_count:
        for at in amino_atoms:
            atoms["ATC_" + at] += amino_count[a] * amino_atom_composition[a][at]
    
    total_atoms = sum(atoms.values())
    if total_atoms > 0:
        for at in atoms:
            atoms[at] = (atoms[at] / total_atoms) * 100
    
    return atoms

def compute_bond_composition(sequence):
    bond_counts = {"Total_bonds": 0, "Hydrogen_bonds": 0, "Single_bonds": 0, "Double_bonds": 0}
    
    for amino_acid in sequence:
        if amino_acid in bonds_data:
            for bond_type in bond_counts:
                bond_counts[bond_type] += bonds_data[amino_acid].get(bond_type, 0)
        else:
            logging.warning(f"Unknown amino acid found in bond computation: {amino_acid}")

    return bond_counts

def compute_feature_comp(df,feature_name, function, *args):

    logging.info(f"Computing feature: {feature_name}")
    
    feature_data = []
    
    for index, row in df.iterrows():
        sequence = row["Sequence"]
        entry = row["Entry"]

        if not callable(function):
            raise TypeError(f"{function} is not a callable function.")
        
        feature_values = function(sequence, *args)
        
        feature_data.append({"Entry": entry, **feature_values})

    feature_df = pd.DataFrame(feature_data)

    
    df_updated = df.merge(feature_df, on="Entry", how="left")

    
    df_updated.to_csv(output_file, index=False)
    logging.info(f"Feature '{feature_name}' added and saved to '{output_file}'.")

    return df_updated

def PCP(sequence):
    
    seq_length = len(sequence)
    counts = Counter(sequence)
    composition = {}
    
    for prop, residues in physio_chemical_properties.items():
        count = sum(counts[aa] for aa in residues if aa in counts)
        composition[prop] = (count / seq_length) * 100 if seq_length > 0 else 0
    
    return composition



def Biopython_Features(sequence):
    try:
        analysis = ProteinAnalysis(sequence)

        return {
            "GRAVY": analysis.gravy(),
            "Aromaticity": analysis.aromaticity(),
            "Isoelectric_Point": analysis.isoelectric_point()
        }
    except Exception as e:
        logging.warning(f"Error computing Biopython features for sequence {sequence[:10]}: {e}")
        return {
            "GRAVY": None,
            "Aromaticity": None,
            "Isoelectric_Point": None
        }


def compute_protein_feature(dataset_path, protein_info_path, out_file):
    global output_file
    global amino_symbol, amino_atoms, amino_atom_composition , bonds_data, physio_chemical_properties
    output_file = out_file

    if not os.path.exists(dataset_path):
        logging.error(f"Error: File '{dataset_path}' not found!")  

    df_source = pd.read_csv(dataset_path)
    df_source.fillna('', inplace=True)

    if not os.path.exists(protein_info_path):
        logging.error(f"Error: File '{protein_info_path}' not found!")

    logging.info("Dataset loaded and NaN values filled.")
        
    protein_data = load_protein_info(protein_info_path)

    amino_symbol = protein_data["amino_symbol"]
    amino_atoms = protein_data["amino_atoms"]
    amino_atom_composition = protein_data["amino_atom_composition"]
    bonds_data = protein_data["bonds_data"]
    physio_chemical_properties = protein_data["physio_chemical_properties"]

    if os.path.exists(output_file):
        logging.info(f"Loading existing file: {output_file}")
        df = pd.read_csv(output_file)
    else:
        logging.info("Starting with dataset CSV file.")
        df = pd.read_csv(dataset_path)

    df = compute_feature_comp(df,"Mono_Amino_Comp_1", PolyP_Comp, 1, 1)
    # df = compute_feature_comp(df,"Di_Amino_Comp_1", PolyP_Comp, 2, 1)
    # df = compute_feature_comp(df,"Di_Amino_Comp_2", PolyP_Comp, 2, 2)
    # df = compute_feature_comp(df,"Di_Amino_Comp_3", PolyP_Comp, 2, 3)

    # df = compute_feature_comp(df,"Tri_Amino_Comp_1", PolyP_Comp, 3, 1)
    # df = compute_feature_comp(df,"Tri_Amino_Comp_2", PolyP_Comp, 3, 2)
    # df = compute_feature_comp(df,"Tetra_Amino_Comp_1", PolyP_Comp, 4, 1)
    # df = compute_feature_comp(df,"Tetra_Amino_Comp_2", PolyP_Comp, 4, 2)
    # df = compute_feature_comp(df,"Penta_Amino_Comp_1", PolyP_Comp, 5, 1)
    # df = compute_feature_comp(df,"Penta_Amino_Comp_2", PolyP_Comp, 5, 2)

    df = compute_feature_comp(df, "Biopython_Features", Biopython_Features)    
    df = compute_feature_comp(df,"ATC_Values", ATC)
    df = compute_feature_comp(df, "Bond_Composition", compute_bond_composition)
    df = compute_feature_comp(df, "physio_chemical_properties", PCP)

    logging.info("Processing complete. All features added iteratively.")
