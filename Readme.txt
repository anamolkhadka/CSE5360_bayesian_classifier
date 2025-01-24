Name: Anamol Khadka
UTA ID: 1001747990
Email: anamol.khadka@mavs.uta.edu

Programming language - Python/Python3
Version - 3.12.4

Hello, please rest all the instructions below before testing the program.


Code Structured and basic information.

1. learn_cpt(training_data_file):
    Reads the training data.
    Counts occurrences for individual and conditional variables.
    Calculates P(B), P(C), P(G|B), and P(F|G,C).

2. printCPT(cpts):
    Prints the calculated Conditional Probability Tables (CPTs) in a formatted way.

3. query_probability(query_vars, evidence_vars, cpts):
    Computes the probability of query variables given evidence using inference by enumeration.

4. enumerate_all(vars, evidence):
    Recursively computes the total probability for all variable combinations.

5. probability_given_parents(var, evidence, cpts):
    Retrieves the probability of a variable given its parents from the CPTs.

6. parse_query(query):
    Parses a query string into query variables and evidence.

7. parse_variables(vars_str):
    Converts query and evidence strings into a structured dictionary format.

8. main():
    Reads training data file and learns CPTs.
    Prints CPTs.
    Provides an interactive query loop to calculate probabilities.
    Exits on Query: None.


How to run the code ?

- Compile the program with following command: python3 bnet.py training_data.txt
- The program will show the conditional probabilities for the variables and ask for the query prompt.
- Write query as: Bf given Cf where B is False and C is False.
- The program will output the enumerated numerator probability, enumerated denominator probability and the final probability values.
- For example:
    Query: Bf given Ff
    Query Probability (Numerator):  0.1257443047739496
    Evidence Probability (Denominator):  0.24375157878381035
    Probability: 0.5159
- Provide command as "None" for exiting the program.
- See the Screenshot for the demo.



Bayes Classifier Assignment.

Two Parts:
1. Given the training data from the text file, learn the conditional probality tables for the bayesian network.
The conditional probabilities should match the probabilities provided in the sample exam bnet.png.

2. The program should implement some Joint Probability Distribution (JPD) to calculate the JPD for any events using the 
conditional probability table and calculate the conditional probability for any given event using inference through enumeration. 
Inference through enumeration will list down all the possible combination of nodes that are not listed for calculating JPD.
For eg: P(B=t, G=f | F=f). Calculate P(B=t, G=f) JPD. Calculate P(F=f) using inference from the previous nodes.


Approach:

Part 1: Learning the conditional probabilities tables (CPTS) for the given Bayesian network from the training data.

Steps to Calculate Probabilities
1. Count Occurrences:
    Iterate through the training data to count the occurrences of:
        - Marginal probabilities: Count occurrences of B, C
        - Conditional probabilities:
            - G|B : Count how many times G is True or False for each value of B.
            - F|G,C : Count how many times F is True or False for each combination of G and C.

2. Normalize to Calculate Probabilities:
    - Marginal probabilities: Divide the count of each variable state by the total number of rows.
    - Conditional probabilities: Divide the count of a state given its parents by the total count of the parent configuration.

Part 2: Calculate conditional probabilities using JPD for any variables.
1. Inference by enumeration. Use bayesian networks inference formula to calculate the conditional probabilities by enumerating all the 
hidden variables and sum over their probabilities based on the network.
2. Parsing the query variables and evidence variables into a format compatible with the network.
3. Use the learned conditional probabilities tables (CPTs) to calculate the Joint Distribution Probabilities (JPD).
4. Normalize results to handle conditional probabilities.
