# Extract features with importance less than -0.001
features_list = []
features_dict = {}

with open('paste.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        if len(parts) == 2:
            feature = parts[1].split(': ')[0].split(' ')[0]
            try:
                importance = float(parts[1].split(': ')[1])
                if importance < -0.001:
                    features_list.append(feature)
                    features_dict[feature] = importance
            except (IndexError, ValueError):
                continue

# Sort the dictionary by importance (most negative first)
sorted_features_dict = dict(sorted(features_dict.items(), key=lambda item: item[1]))

print("Features List:")
print(features_list)
print("\nFeatures Dictionary:")
print(sorted_features_dict)