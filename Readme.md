# Bayesian Network Classifier

## Overview

This project implements a **Bayesian Network Classifier** that learns conditional probability tables (CPTs) from training data and performs inference using enumeration. The classifier is built to model the relationships between the following variables:

- **B:** Whether there is a baseball game on TV (True/False)
- **G:** Whether George watches TV (True/False)
- **C:** Whether George is out of cat food (True/False)
- **F:** Whether George feeds his cat (True/False)

The system processes a dataset containing these variables' historical occurrences and uses **Bayesian Inference** to answer probabilistic queries.

---

## Project Objective

The project is divided into two main parts:

### **Part 1: Learning Conditional Probability Tables (CPTs)**

The goal is to calculate the conditional probabilities from the given training data file. The following probabilities are computed based on the data:

1. **Marginal Probabilities:**  
   - `P(B)`: Probability of baseball game on TV.  
   - `P(C)`: Probability of George being out of cat food.

2. **Conditional Probabilities:**  
   - `P(G | B)`: Probability of George watching TV given whether a baseball game is on.  
   - `P(F | G, C)`: Probability of George feeding his cat given whether he watched TV and has cat food.

**Approach to Learning:**  
1. Read the training data to count occurrences of events and their combinations.
2. Normalize counts to compute probabilities.
3. Use Laplace smoothing to avoid zero-probability issues.

---

### **Part 2: Inference by Enumeration**

The system calculates conditional probabilities for any events using **Inference by Enumeration**, which involves:

1. Parsing the user's query input to extract target variables and evidence.
2. Computing the **Joint Probability Distribution (JPD)** for the query variables.
3. Normalizing the result to obtain conditional probabilities based on available evidence.

**Example Queries:**
- `Query: Bt Gf given Ff` → Computes `P(B = True, G = False | F = False)`
- `Query: Bt Ff` → Computes `P(B = True, F = False)`

**Inference Process:**
- Enumerate all possible values of hidden variables.
- Use learned CPTs to compute probabilities.
- Normalize the result to provide the final conditional probability.

---

## How to Run the Project

1. **Install Python:** Ensure Python 3.12.4 or higher is installed.
2. **Run the program using the following command:**
   ```bash
   python3 bnet.py training_data.txt
