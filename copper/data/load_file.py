import json


def load_json_file(file_path):
    """
    Load and print the content of a JSON file.
    Parameters:
    file_path (str): The path to the JSON file.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            print(data[1])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print("Error decoding JSON from the file.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    path = "./copper/data/DX_oat.json"
    load_json_file(path)
