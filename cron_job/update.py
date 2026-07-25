import os

def add_for_loop_to_file(folder_path, filename, loop_variable="i", iterable="range(10)", loop_body="    print(i)"):
    """
    Opens an existing Python file in the given folder and appends a basic for loop.

    Args:
        folder_path (str): Path to the folder containing the file.
        filename (str): Name of the Python file (e.g., 'script.py').
        loop_variable (str): Name of the loop variable.
        iterable (str): The iterable expression (e.g., 'range(10)').
        loop_body (str): The body of the loop (must include correct indentation).
    """
    file_path = os.path.join(folder_path, filename)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    for_loop_code = f"\nfor {loop_variable} in {iterable}:\n{loop_body}\n"

    with open(file_path, "a") as f:
        f.write(for_loop_code)

    print(f"For loop appended to {file_path}")



def main():
    try:
        add_for_loop_to_file(
            folder_path="../text_scanner_project",
            filename="ImageFormat.py",
            loop_variable="i",
            iterable="range(5)",
            loop_body="    print(i)"
)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()