
import json

def categorize_data(file_path):

    categorized_data = {}
    current_values = []
    current_key = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()

                if line.startswith('`') and line.endswith('`'):
                    if current_key is not None:
                        categorized_data[current_key] = current_values
                    current_key = line[1:-1]
                    current_values = []
                else:
                    if current_key is not None:
                        current_values.append(line)

        if current_key is not None:
            categorized_data[current_key] = current_values

    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return categorized_data

file_path = './descriptorsMap/descriptorsMapping.txt'
result = categorize_data(file_path)
print(result)


if __name__ == '__main__':

    with open('./descriptorsMap/descriptorsMapping.json', 'r') as f:
        result = json.load(f)

    number_of_keys = len(result)
    print(f"Number of keys: {number_of_keys}")

    total_number_of_values = sum(len(value) for value in result.values())
    print(f"Total number of values: {total_number_of_values}")