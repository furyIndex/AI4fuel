import pubchempy as pcp
import pandas as pd



def getSmilesFromCAS(file_path):

    df = pd.read_excel(file_path, sheet_name='Sheet1')
    smiles_list = []

    cas_data = df['CAS'].values.tolist()
    cas_list = []

    for cas in cas_data:
        if type(cas) is str:
            cas_list.append(cas)
        else:
            print(cas_data.index(cas))
    print("Effective CAS sample： ", len(cas_list))
    i = 2
    for cas in cas_list:
        print(i)
        print(cas)
        compounds = pcp.get_compounds(cas, 'name')
        print(len(compounds))
        if len(compounds) == 1:
            compound = compounds[0]
            print(compound.isomeric_smiles)

            if compound.isomeric_smiles is None:
                smiles_list.append("null")
                print("null")
            else:
                smiles_list.append(compound.isomeric_smiles)
        if len(compounds) == 0:
            smiles_list.append("null")
            print("null")

        if len(compounds) > 1:
            smiles_list.append("multiple")
        print("smiles length：{}".format(len(smiles_list) + 1))
        print("---------------------------------")
        i = i+1
    print("========================================")

    print(len(smiles_list))
    df = pd.DataFrame({'smiles_list' : smiles_list})

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:

        df.to_excel(writer, sheet_name="smiles", index=False)





if __name__ == '__main__':
    file_path = "../data/MON.xlsx"

    getSmilesFromCAS(file_path)