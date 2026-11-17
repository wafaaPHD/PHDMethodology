def remove_words_from_file(input_file, output_file, words_to_remove):
    try:
        # Open the input file for reading
        with open(input_file, 'r') as file_in:
            # Read the entire content of the file
            file_content = file_in.read()

        # Replace words to remove with empty string
        for word in words_to_remove:
            file_content = file_content.replace(word, ' ')
            file_content = file_content.replace('[ ', '[')


        # Open the output file for writing
        with open(output_file, 'w') as file_out:
            # Write the modified content back to the output file
            file_out.write(file_content)

        print(f"Words removed successfully from '{input_file}'.")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
input_filename = 'output_pos.txt'
output_filename = 'output_det_remove.txt'
words_to_remove = [' the ', ' is ', ' on ', ' a ','The ','Is ','On ','A ',' its ',' an ','Its ']

remove_words_from_file(input_filename, output_filename, words_to_remove)
