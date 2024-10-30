import pandas as pd
from typing import List, Tuple, Dict, Set
from itertools import combinations




def read_data(file_path: str) -> pd.DataFrame:
   print(f"Reading data from {file_path}...")
   df = pd.read_excel(file_path)
   print(f"Data read successfully. Shape: {df.shape}")
   return df


def parse_functional_dependencies() -> List[Tuple[List[str], List[str]]]:
    """
    Parse functional dependencies from user input.
    
    Returns:
        List[FunctionalDependency]: A list of parsed functional dependencies.
    """
    fds = []
    print("Enter functional dependencies one by one in the format 'A, B -> C, D'. Type 'done' when finished:")
    
    while True:
        user_input = input("FD: ")
        if user_input.lower() == "done":
            break
        
        try:
            # Split at '->' to separate determinant and dependent attributes
            left_side, right_side = user_input.split("->")
            left_attributes = [attr.strip() for attr in left_side.split(",")]
            right_attributes = [attr.strip() for attr in right_side.split(",")]
            
            # Adds the functional dependency into the list of FDs 
            fds.append((left_attributes, right_attributes))
            
        except ValueError:
            print("Invalid format. Please use 'A, B -> C, D' format.")


# def read_fds(file_path: str) -> List[Tuple[List[str], List[str]]]:
#    print(f"Reading functional dependencies from {file_path}...")
#    with open(file_path, 'r') as file:
#        lines = file.readlines()
#    fds = []
#    for line in lines:
#        if '->' in line:
#            parts = line.strip().split('->')
#            lhs = [attr.strip() for attr in parts[0].split(',')]
#            rhs = [attr.strip() for attr in parts[1].split(',')]
#            fds.append((lhs, rhs))
#    print(f"Functional dependencies read successfully: {fds}")
#    return fds




def find_mvds(data: pd.DataFrame) -> List[Tuple[List[str], List[str]]]:
    mvds = []
    columns = data.columns

    for lhs_size in range(1, len(columns)):
        for lhs_comb in combinations(columns, lhs_size):
            lhs_comb = list(lhs_comb)
            for rhs_comb in columns:
                if rhs_comb not in lhs_comb:
                    lhs_values = data.groupby(lhs_comb)[rhs_comb].nunique()
                    if all(lhs_values > 1):
                        mvds.append((lhs_comb, [rhs_comb]))

    return mvds







def calculate_closure(attributes: Set[str], fds: List[Tuple[List[str], List[str]]]) -> Set[str]:
   closure = set(attributes)
   while True:
       new_attributes = closure.copy()
       for lhs, rhs in fds:
           if set(lhs).issubset(closure):
               new_attributes.update(rhs)
       if new_attributes == closure:
           break
       closure = new_attributes
   return closure




def identify_multivalued_attributes(data: pd.DataFrame) -> Tuple[List[str], Dict[str, List[Tuple[int, str]]]]:
   mv_attributes = []
   violations = {}
   

   for column in data.columns:
       multi_valued_indices = []
       for idx, value in data[column].items():
           if isinstance(value, str) and ',' in value:
               multi_valued_indices.append((idx, value))
       if multi_valued_indices:
           mv_attributes.append(column)
           violations[column] = multi_valued_indices


   return mv_attributes, violations




def normalize_to_1nf(data: pd.DataFrame, primary_keys: List[List[str]]) -> List[Tuple[pd.DataFrame, List[str]]]:
   mv_attributes, violations = identify_multivalued_attributes(data)


   # Print violations
   if violations:
       print("1NF Violations found:")
       for attr, vals in violations.items():
           print(f"Attribute '{attr}' has multiple values in a single field:")
           for idx, val in vals:
               print(f"  Row {idx}: {val}")
       print()


   base_table = data.drop(columns=mv_attributes, errors='ignore')
   tables = [(base_table, primary_keys[0])]
   for mv_attr in mv_attributes:
       for key in primary_keys:
           exploded_table = data[list(key) + [mv_attr]].copy()
           exploded_table[mv_attr] = exploded_table[mv_attr].apply(lambda x: x.split(',') if isinstance(x, str) else x)
           exploded_table = exploded_table.explode(mv_attr).dropna().drop_duplicates()
           tables.append((exploded_table, key + [mv_attr]))

    

   return tables



def normalize_to_2nf(tables: List[Tuple[pd.DataFrame, List[str]]], fds: List[Tuple[List[str], List[str]]]) -> Tuple[List[Tuple[pd.DataFrame, List[str]]], List[Tuple[List[str], List[str]]]]:
    print("Normalizing to 2NF...")
    if not fds:
        print("No functional dependencies provided.")
        return tables, fds

    new_tables = []
    updated_fds = fds.copy()
    print("TABLES", tables)
    print("UPDATED_FDS", updated_fds)

    for table, primary_key in tables:
        partial_dependencies = []
        for lhs, rhs in updated_fds:
            # Check if LHS is a proper subset of the primary key
            if set(lhs).issubset(primary_key) and not set(lhs) == set(primary_key):
                # Exclude one-to-one dependencies unless RHS is part of the primary key
                if len(lhs) == 1 and len(rhs) == 1 and rhs[0] not in primary_key:
                    continue
                partial_dependencies.append((lhs, rhs))
            

        print("Identified partial dependencies for table with PK {}:".format(primary_key))
        for pd in partial_dependencies:
            print(f"LHS: {pd[0]}, RHS: {pd[1]}")

        # Create tables for each partial dependency
        for lhs, rhs in partial_dependencies:
            columns_in_df = [col for col in (lhs + rhs) if col in table.columns]
            new_table = table[columns_in_df].drop_duplicates()
            new_tables.append((new_table, lhs))
            
            # Drop the columns in rhs from the original table
            table = table.drop(columns=[col for col in rhs if col in table.columns]).drop_duplicates()
            
            print(f"Attempting to remove FD: ({lhs}, {rhs})")
            try:
                updated_fds.remove((lhs, rhs))
            except ValueError as e:
                print(f"Error: {e}. The FD ({lhs}, {rhs}) was not found in updated_fds.")

        # Add the modified original table back to the list of new tables
        remaining_columns = list(set(table.columns) - set(primary_key))
        remaining_columns = [col for col in remaining_columns if col in table.columns]  # Ensure columns are still in the table
        if remaining_columns:
            remaining_table = table[primary_key + remaining_columns].drop_duplicates()
            new_tables.append((remaining_table, primary_key))
        else:
            new_tables.append((table, primary_key))

    print(f"Functional Dependencies after 2NF: {updated_fds}")
    return new_tables, updated_fds






def normalize_to_3nf(tables: List[Tuple[pd.DataFrame, List[str]]], fds: List[Tuple[List[str], List[str]]]) -> Tuple[List[Tuple[pd.DataFrame, List[str]]], List[Tuple[List[str], List[str]]]]:
   print("Normalizing to 3NF...")
   if not fds:
       print("No functional dependencies provided.")
       return tables, fds


   print(f"Functional Dependencies before 3NF: {fds}")


   transitive_dependencies = []


   # Identify transitive dependencies
   for lhs, rhs in fds:
       for table, primary_key in tables:
           if set(lhs).issubset(table.columns) and not set(lhs).issubset(primary_key):
               # Ensure the transitive dependency condition
               if not any(attr in primary_key for attr in lhs) and not any(attr in primary_key for attr in rhs):
                   transitive_dependencies.append((lhs, rhs))


   print("Identified transitive dependencies:")
   for td in transitive_dependencies:
       print(f"LHS: {td[0]}, RHS: {td[1]}")


   new_tables = []
   updated_fds = fds.copy()


   for table, primary_key in tables:
       columns = set(table.columns)
       table_fds = [fd for fd in fds if set(fd[0]).issubset(columns) and set(fd[1]).issubset(columns)]


       # Create tables for each transitive dependency
       for lhs, rhs in table_fds:
           if (lhs, rhs) in transitive_dependencies:
               new_table = table[lhs + rhs].drop_duplicates()
               new_tables.append((new_table, lhs))
               for col in rhs:
                   if col in table.columns:
                       table = table.drop(columns=[col])
               updated_fds.remove((lhs, rhs))


       new_tables.append((table, primary_key))


   print(f"Functional Dependencies after 3NF: {updated_fds}")
   return new_tables, updated_fds








def normalize_to_bcnf(tables: List[Tuple[pd.DataFrame, List[str]]], fds: List[Tuple[List[str], List[str]]], primary_keys: List[List[str]]) -> Tuple[List[Tuple[pd.DataFrame, List[str]]], List[Tuple[List[str], List[str]]]]:
   print("Normalizing to BCNF...")
   if not fds:
       print("No functional dependencies provided.")
       return tables, fds


   def is_superkey(attrs: List[str], primary_keys: List[List[str]]) -> bool:
       for key in primary_keys:
           if set(key).issubset(set(attrs)):
               return True
       return False


   print(f"Functional Dependencies before BCNF: {fds}")
   print(f"Primary Keys before BCNF: {primary_keys}")


   new_tables = []
   updated_fds = fds.copy()
   bcnf_violations = []


   for table, primary_key in tables:
       if not isinstance(table, pd.DataFrame):
           print(f"Error: Expected DataFrame but got {type(table)}")
           continue


       columns = set(table.columns)
       table_fds = [fd for fd in fds if set(fd[0]).issubset(columns) and set(fd[1]).issubset(columns)]


       for lhs, rhs in table_fds:
           if not is_superkey(lhs, primary_keys):
               bcnf_violations.append((lhs, rhs))


               new_primary_key = lhs
               if not any(set(new_primary_key).issubset(set(key)) for key in primary_keys):
                   primary_keys.append(new_primary_key)


               new_table = table[lhs + rhs].drop_duplicates()
               new_tables.append((new_table, lhs))
               for col in rhs:
                   if col in table.columns:
                       table = table.drop(columns=[col])
               updated_fds.remove((lhs, rhs))


       new_tables.append((table, primary_key))


   print("Identified BCNF violations:")
   for violation in bcnf_violations:
       print(f"LHS: {violation[0]}, RHS: {violation[1]}")


   print(f"Functional Dependencies after BCNF: {updated_fds}")
   print(f"Primary Keys after BCNF: {primary_keys}")
   return new_tables, updated_fds




def normalize_to_4nf(tables: List[Tuple[pd.DataFrame, List[str]]], mvds: List[Tuple[List[str], List[str]]], primary_keys: List[List[str]]) -> List[Tuple[pd.DataFrame, List[str]]]:
   print("Normalizing to 4NF...")
   if not mvds:
       print("No multi-valued dependencies provided.")
       return tables


   new_tables = []
   for table, primary_key in tables:
       columns = set(table.columns)
       for lhs, rhs in mvds:
           if set(rhs).issubset(columns):
               if not any(set(lhs).issubset(set(key)) for key in primary_keys):
                   new_table = table[lhs + rhs].drop_duplicates()
                   new_tables.append((new_table, lhs))
                   for col in rhs:
                       if col in table.columns:
                           table = table.drop(columns([col]))
       new_tables.append((table, primary_key))
   return new_tables




def normalize_to_5nf(tables: List[Tuple[pd.DataFrame, List[str]]], primary_keys: List[List[str]]) -> List[Tuple[pd.DataFrame, List[str]]]:
   print("Normalizing to 5NF...")


   def join_dependency_violated(df: pd.DataFrame, primary_keys: List[List[str]]) -> bool:
       for primary_key in primary_keys:
           non_key_attrs = list(set(df.columns) - set(primary_key))
           for i in range(1, len(non_key_attrs) + 1):
               for subset in combinations(non_key_attrs, i):
                   lhs = list(primary_key) + list(subset)
                   rhs = list(set(df.columns) - set(lhs))
                   if set(lhs).issubset(df.columns) and set(rhs).issubset(df.columns):
                       if df[lhs + rhs].drop_duplicates().shape[0] != df.drop_duplicates().shape[0]:
                           return True
       return False


   new_tables = []
   for table, primary_key in tables:
       if join_dependency_violated(table, primary_keys):
           for primary_key in primary_keys:
               non_key_attrs = list(set(table.columns) - set(primary_key))
               for i in range(1, len(non_key_attrs) + 1):
                   for subset in combinations(non_key_attrs, i):
                       lhs = list(primary_key) + list(subset)
                       rhs = list(set(table.columns) - set(lhs))
                       if set(lhs).issubset(table.columns) and set(rhs).issubset(table.columns):
                           if table[lhs + rhs].drop_duplicates().shape[0] != table.drop_duplicates().shape[0]:
                               new_table_lhs = table[lhs].drop_duplicates()
                               new_table_rhs = table[rhs].drop_duplicates()
                               new_tables.append((new_table_lhs, primary_key))
                               new_tables.append((new_table_rhs, rhs))
                               break
                   else:
                       continue
                   break
           else:
               new_tables.append((table, primary_key))
       else:
           new_tables.append((table, primary_key))


   return new_tables




def generate_schema(df_list: List[Tuple[pd.DataFrame, List[str]]], primary_keys: List[List[str]]) -> str:
   schema = ""
   for i, (df, pk) in enumerate(df_list):
       if isinstance(df, pd.DataFrame):
           table_name = f"Table_{i+1}"
           columns = ", ".join(df.columns)
           pk_str = ", ".join(pk)


           reordered_columns = [col for col in df.columns if col in pk] + \
                               [col for col in df.columns if col not in pk]
           columns = ", ".join(reordered_columns)


           schema += f"{table_name} ({columns})\n"
           schema += f"PK: {pk_str}\n\n"
       else:
           print(f"Error: Expected DataFrame but got {type(df)}")
   return schema




def main():
    data_file = 'testinput.xlsx'
    # fds_file = 'fds1.txt'

    df = read_data(data_file)
    print(f"Initial data: \n{df.head()}\n")

    fds = parse_functional_dependencies()
    print("Functional Dependencies Processed")

    # Get primary keys from the user
    primary_keys_input = input("Enter the primary keys (can be composite, separated by commas; multiple keys separated by semicolons): ")
    primary_keys = [key.strip().split(',') for key in primary_keys_input.split(';')]
    print(f"Provided Primary Keys: {primary_keys}\n")

    mvds = find_mvds(df)
    print(f"Identified Multi-valued Dependencies: {mvds}\n")

    highest_normal_form = input("Enter the highest normalization form (1NF, 2NF, 3NF, BCNF, 4NF, 5NF): ").upper()

    # Normalize to 1NF
    tables = normalize_to_1nf(df, primary_keys)

    if highest_normal_form == "1NF":
        schema = generate_schema(tables, primary_keys)
        print("Normalized Database Schema:")
        print(schema)
        return

    # Normalize to higher normal forms based on user input
    if highest_normal_form in ["2NF", "3NF", "BCNF", "4NF", "5NF"]:
        tables, fds = normalize_to_2nf(tables, fds)
        print(f"Primary Keys after 2NF: {[(pk, list(t.columns)) for t, pk in tables]}\n")

    if highest_normal_form in ["3NF", "BCNF", "4NF", "5NF"]:
        tables, fds = normalize_to_3nf(tables, fds)
        print(f"Primary Keys after 3NF: {[(pk, list(t.columns)) for t, pk in tables]}\n")

    if highest_normal_form in ["BCNF", "4NF", "5NF"]:
        tables, fds = normalize_to_bcnf(tables, fds, primary_keys)
        print(f"Primary Keys after BCNF: {primary_keys}\n")

    if highest_normal_form in ["4NF", "5NF"]:
        tables = normalize_to_4nf(tables, mvds, primary_keys)

    if highest_normal_form == "5NF":
        tables = normalize_to_5nf(tables, primary_keys)

    schema = generate_schema(tables, primary_keys)
    print("Normalized Database Schema:")
    print(schema)

if __name__ == "__main__":
    main()