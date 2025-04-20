import itertools

# Get itemset support count
def get_item_support(transactions, itemsets):
    support_count = {}
    for itemset in itemsets:
        for transaction in transactions:
            if itemset.issubset(transaction):
                support_count[itemset] = support_count.get(itemset, 0) + 1
    return support_count

# Generate frequent itemsets
def apriori(transactions, min_support):
    transactions = list(map(set, transactions))
    itemsets = []
    support_data = {}

    # Create initial 1-itemsets
    items = set(itertools.chain.from_iterable(transactions))
    current_itemsets = [frozenset([item]) for item in items]
    
    k = 1
    while current_itemsets:
        item_support = get_item_support(transactions, current_itemsets)
        total_transactions = len(transactions)

        # Filter based on support threshold
        frequent_itemsets = []
        for itemset, count in item_support.items():
            support = count / total_transactions
            if support >= min_support:
                frequent_itemsets.append(itemset)
                support_data[itemset] = support
        
        itemsets.extend(frequent_itemsets)
        
        # Generate next itemsets of length k+1
        current_itemsets = [
            i.union(j) for i in frequent_itemsets for j in frequent_itemsets
            if len(i.union(j)) == k + 1
        ]
        current_itemsets = list(set(current_itemsets))
        k += 1
    
    return itemsets, support_data

# Generate association rules
def generate_rules(frequent_itemsets, support_data, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) >= 2:
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if antecedent in support_data and consequent:
                        confidence = support_data[itemset] / support_data[antecedent]
                        if confidence >= min_confidence:
                            rules.append((antecedent, consequent, support_data[itemset], confidence))
    return rules

# Example usage
transactions = [
    ['a', 'b', 'c'],
    ['b', 'd'],
    ['b', 'a', 'd', 'c'],
    ['e', 'd'],
    ['a', 'b', 'c', 'd'],
    ['f'],
]

min_support = 0.5
min_confidence = 0.7

frequent_itemsets, support_data = apriori(transactions, min_support)
rules = generate_rules(frequent_itemsets, support_data, min_confidence)

# Output
print("Frequent Itemsets:")
for itemset in frequent_itemsets:
    print(set(itemset), "=>", f"support: {support_data[itemset]:.2f}")

print("\n\nAssociation Rules:")
for antecedent, consequent, support, confidence in rules:
    print(f"{set(antecedent)} => {set(consequent)} (support: {support:.2f}, confidence: {confidence:.2f})")
