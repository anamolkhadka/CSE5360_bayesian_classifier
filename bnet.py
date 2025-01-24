"""
Bayes Classifier Assignment.
Two Parts:
1. Given the training data from the text file, learn the conditional probality tables for the bayesian network.
The conditional probabilities should match the probabilities provided in the sample exam bnet.png.

2. The program should implement Joint Probability Distribution (JPD) to calculate the JPD for any events using the 
conditional probability table and calculate the conditional probability for any given event using inference through enumeration. 
Inference through enumeration will list down all the possible combination of nodes that are not listed for calculating JPD.
For eg: P(B=t, G=f | F=f). Calculate P(B=t, G=f) JPD. Calculate P(F=f) using inference from the previous nodes.
"""

import sys
from collections import defaultdict

def learn_cpt(training_data_file):
    # Initialize counts
    counts = defaultdict(int)
    total_rows = 0

    # Read the training data
    with open(training_data_file, 'r') as file:
        for line in file:
            total_rows += 1
            b, g, c, f = map(int, line.strip().split())

            # Increment counts for individual variables
            counts[f"B={b}"] += 1
            counts[f"C={c}"] += 1

            # Increment counts for conditional probabilities
            counts[f"G={g}|B={b}"] += 1
            counts[f"F={f}|G={g},C={c}"] += 1

            # Update joint counts for normalization
            counts[f"G={g},C={c}"] += 1

    # Calculate probabilities
    prob_b = {key: val / total_rows for key, val in counts.items() if key.startswith("B=")}
    prob_c = {key: val / total_rows for key, val in counts.items() if key.startswith("C=")}

    prob_g_given_b = {
        key: val / counts[f"B={key[-1]}"]
        for key, val in counts.items() if key.startswith("G=") and "|B=" in key
    }

    prob_f_given_g_c = {}
    # Iterate over all possible combinations of F, G, C
    for f in ["0", "1"]:
        for g in ["0", "1"]:
            for c in ["0", "1"]:
                key = f"F={f}|G={g},C={c}"
                parent_key = f"G={g},C={c}"
                current_count = counts[key]
                parent_count = counts[parent_key]

                # Avoid division by zero
                prob_f_given_g_c[key] = current_count / parent_count if parent_count > 0 else 0


    # Store results
    cpts = {
        "P(B)": prob_b,
        "P(C)": prob_c,
        "P(G|B)": prob_g_given_b,
        "P(F|G,C)": prob_f_given_g_c,
    }

    return cpts

def printCPT(cpts):
    # Display the probabilities
    for cpt_name, probabilities in cpts.items():
        print(f"{cpt_name}:")
        for key, prob in probabilities.items():
            print(f"  {key}: {prob:.4f}")
        print()

def query_probability(query_vars, evidence_vars, cpts):
    def enumerate_all(vars, evidence):
        if not vars:
            return 1.0

        first, rest = vars[0], vars[1:]
        if first in evidence:
            prob = probability_given_parents(first, evidence, cpts)
            return prob * enumerate_all(rest, evidence)
        else:
            total_prob = 0
            for value in [0, 1]:
                new_evidence = evidence.copy()
                new_evidence[first] = value
                prob = probability_given_parents(first, new_evidence, cpts)
                total_prob += prob * enumerate_all(rest, new_evidence)
            return total_prob

    # Normalize the query using evidence
    all_vars = ["B", "G", "C", "F"]
    hidden_vars = [var for var in all_vars if var not in query_vars and var not in evidence_vars]
    
    # Combine query and evidence
    query_evidence = {**query_vars, **evidence_vars}
    query_prob = enumerate_all(all_vars, query_evidence)
    evidence_prob = enumerate_all(all_vars, evidence_vars)

    print("Query Probability (Numerator): ", query_prob)
    print("Evidence Probability (Denominator): ", evidence_prob)

    return query_prob / evidence_prob if evidence_prob != 0 else 0

def probability_given_parents(var, evidence, cpts):
    if var == "B":
        return cpts["P(B)"][f"B={evidence[var]}"]
    elif var == "C":
        return cpts["P(C)"][f"C={evidence[var]}"]
    elif var == "G":
        parent = f"B={evidence['B']}"
        return cpts["P(G|B)"][f"G={evidence[var]}|{parent}"]
    elif var == "F":
        parent = f"G={evidence['G']},C={evidence['C']}"
        return cpts["P(F|G,C)"][f"F={evidence[var]}|{parent}"]
    else:
        raise ValueError(f"Unknown variable: {var}")

def parse_query(query):
    # Parse the query string
    if "given" in query:
        query_part, evidence_part = query.split("given")
        query_vars = parse_variables(query_part.strip())
        evidence_vars = parse_variables(evidence_part.strip())
    else:
        query_vars = parse_variables(query.strip())
        evidence_vars = {}
    return query_vars, evidence_vars

def parse_variables(vars_str):
    # Parse variables like "Bt Gf" into {"B": 1, "G": 0}
    vars_map = {}
    for var in vars_str.split():
        var_name, var_value = var[0], var[1]
        vars_map[var_name] = 1 if var_value == "t" else 0
    return vars_map

def main():
    # Command-line argument handling
    if len(sys.argv) < 2:
        print("Usage: python bnet.py <training_data>")
        sys.exit(1)

    training_data_file = sys.argv[1]

    # Learn the CPTs
    cpts = learn_cpt(training_data_file)

    # Print the Conditional Probability Table.
    printCPT(cpts)
    
    # Query loop
    while True:
        query = input("Query: ").strip()
        if query.lower() == "none":
            break
        query_vars, evidence_vars = parse_query(query)
        prob = query_probability(query_vars, evidence_vars, cpts)
        print(f"Probability: {prob:.4f}")

if __name__ == "__main__":
    main()