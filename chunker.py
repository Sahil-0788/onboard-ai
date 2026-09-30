import os

# ── SETTINGS ──────────────────────────────────────────────
# Which file types count as "code" — we'll only read these
CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".html", ".css"}

# Folders to skip entirely (no point reading these — not real project code)
SKIP_FOLDERS = {"node_modules", ".git", "venv", "__pycache__", "dist", "build"}

# How many lines of code go into one "chunk"
CHUNK_SIZE = 40
# ──────────────────────────────────────────────────────────


def find_code_files(root_folder):
    """
    Walks through every folder/subfolder inside root_folder,
    and returns a list of full paths to every code file found,
    skipping folders we don't care about.
    """
    code_files = []

    # os.walk goes through the folder tree: current folder, its subfolders, its files
    for current_folder, subfolders, files in os.walk(root_folder):
        # Remove skip-folders from subfolders list so os.walk doesn't enter them
        subfolders[:] = [f for f in subfolders if f not in SKIP_FOLDERS]

        for filename in files:
            # os.path.splitext splits "app.py" into ("app", ".py")
            _, extension = os.path.splitext(filename)
            if extension in CODE_EXTENSIONS:
                full_path = os.path.join(current_folder, filename)
                code_files.append(full_path)

    return code_files


def chunk_file(file_path):
    """
    Reads one file and splits its contents into chunks of CHUNK_SIZE lines each.
    Returns a list of chunk dictionaries with useful info attached.
    """
    chunks = []

    # encoding="utf-8", errors="ignore" avoids crashing on weird/foreign characters
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Walk through the lines, CHUNK_SIZE at a time
    for i in range(0, len(lines), CHUNK_SIZE):
        chunk_lines = lines[i : i + CHUNK_SIZE]
        chunk_text = "".join(chunk_lines)

        if chunk_text.strip():  # skip empty chunks (blank sections of a file)
            filename_only = os.path.basename(file_path)
            # Prepend the filename into the text itself, so the embedding
            # "knows" which file this chunk belongs to, not just the raw code.
            text_with_context = f"# File: {filename_only}\n{chunk_text}"

            chunks.append({
                "file": file_path,
                "start_line": i + 1,
                "end_line": i + len(chunk_lines),
                "text": text_with_context,
            })

    return chunks


def main():
    folder = input("Enter the full path to the codebase folder: ").strip()

    if not os.path.isdir(folder):
        print("That folder doesn't exist. Double check the path.")
        return

    files = find_code_files(folder)
    print(f"\nFound {len(files)} code file(s).\n")

    all_chunks = []
    for file_path in files:
        chunks = chunk_file(file_path)
        all_chunks.extend(chunks)
        print(f"  {file_path} → {len(chunks)} chunk(s)")

    print(f"\nTotal chunks across whole codebase: {len(all_chunks)}")


if __name__ == "__main__":
    main()