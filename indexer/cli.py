import argparse

from indexer.repo_parser import RepoParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a repo-level code chunk index using tree-sitter.")
    parser.add_argument("repo_path", help="Path to the repository to index")
    parser.add_argument("-o", "--output", default="repo_index.json", help="Output JSON path")
    args = parser.parse_args()

    repo_parser = RepoParser(args.repo_path)
    chunks = repo_parser.parse_repo()
    repo_parser.save_index(chunks, args.output)
    print(f"Indexed {len(chunks)} chunks from {args.repo_path} -> {args.output}")


if __name__ == "__main__":
    main()
