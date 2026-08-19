from typing import List, Optional

from tree_sitter import Node, Tree

from indexer.languages.base import BaseExtractor
from indexer.models import Chunk


def _node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


class PythonExtractor(BaseExtractor):
    language_name = "python"

    def extract_chunks(self, tree: Tree, source_bytes: bytes, file_path: str) -> List[Chunk]:
        root = tree.root_node
        imports = self._extract_imports(root, source_bytes)
        chunks: List[Chunk] = []
        self._walk(root, source_bytes, file_path, imports, class_name=None, chunks=chunks)
        return chunks

    # -- traversal -----------------------------------------------------

    def _walk(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        imports: List[str],
        class_name: Optional[str],
        chunks: List[Chunk],
    ) -> None:
        for child in node.children:
            definition, span_node = self._unwrap_decorated(child)

            if definition.type == "class_definition":
                chunks.append(
                    self._make_chunk(
                        node=definition,
                        span_node=span_node,
                        source_bytes=source_bytes,
                        file_path=file_path,
                        imports=imports,
                        chunk_type="class",
                        class_name=None,
                    )
                )
                body = definition.child_by_field_name("body")
                new_class_name = self._get_name(definition, source_bytes)
                if body is not None:
                    self._walk(body, source_bytes, file_path, imports, new_class_name, chunks)
                continue

            if definition.type == "function_definition":
                is_method = class_name is not None
                chunks.append(
                    self._make_chunk(
                        node=definition,
                        span_node=span_node,
                        source_bytes=source_bytes,
                        file_path=file_path,
                        imports=imports,
                        chunk_type="method" if is_method else "function",
                        class_name=class_name if is_method else None,
                    )
                )
                body = definition.child_by_field_name("body")
                if body is not None:
                    # Functions nested inside this one are plain "function" chunks,
                    # even if this function is itself a method.
                    self._walk(body, source_bytes, file_path, imports, class_name=None, chunks=chunks)
                continue

            self._walk(child, source_bytes, file_path, imports, class_name, chunks)

    @staticmethod
    def _unwrap_decorated(node: Node):
        """Return (definition_node, node_covering_full_source_including_decorators)."""
        if node.type == "decorated_definition":
            inner = node.child_by_field_name("definition")
            return inner, node
        return node, node

    # -- chunk construction ----------------------------------------------

    def _make_chunk(
        self,
        node: Node,
        span_node: Node,
        source_bytes: bytes,
        file_path: str,
        imports: List[str],
        chunk_type: str,
        class_name: Optional[str],
    ) -> Chunk:
        name = self._get_name(node, source_bytes)
        body = node.child_by_field_name("body")
        signature = self._get_signature(node, body, source_bytes)
        docstring = self._get_docstring(body, source_bytes) if body is not None else None
        source_code = _node_text(span_node, source_bytes)
        start_line = span_node.start_point[0] + 1
        end_line = span_node.end_point[0] + 1
        qualified_name = f"{class_name}.{name}" if class_name else name

        return Chunk(
            chunk_id=self._make_chunk_id(file_path, chunk_type, qualified_name, start_line, end_line),
            file_path=file_path,
            language=self.language_name,
            type=chunk_type,
            name=name,
            class_name=class_name,
            signature=signature,
            docstring=docstring,
            imports=list(imports),
            source_code=source_code,
            start_line=start_line,
            end_line=end_line,
        )

    @staticmethod
    def _get_name(node: Node, source_bytes: bytes) -> str:
        name_node = node.child_by_field_name("name")
        return _node_text(name_node, source_bytes) if name_node is not None else ""

    @staticmethod
    def _get_signature(node: Node, body: Optional[Node], source_bytes: bytes) -> str:
        end_byte = body.start_byte if body is not None else node.end_byte
        header = source_bytes[node.start_byte : end_byte].decode("utf-8", errors="replace")
        return header.strip()

    @staticmethod
    def _get_docstring(body: Node, source_bytes: bytes) -> Optional[str]:
        for child in body.children:
            if child.type == "comment":
                continue
            if child.type != "expression_statement":
                return None
            if child.child_count == 0 or child.children[0].type != "string":
                return None
            string_node = child.children[0]
            content_parts = [
                _node_text(c, source_bytes) for c in string_node.children if c.type == "string_content"
            ]
            if content_parts:
                return "".join(content_parts).strip()
            # Fallback for grammar versions without string_content sub-nodes.
            raw = _node_text(string_node, source_bytes).strip()
            for quote in ('"""', "'''", '"', "'"):
                if raw.startswith(quote) and raw.endswith(quote) and len(raw) >= 2 * len(quote):
                    return raw[len(quote) : -len(quote)].strip()
            return raw
        return None

    @staticmethod
    def _make_chunk_id(file_path: str, chunk_type: str, qualified_name: str, start_line: int, end_line: int) -> str:
        # Readable and still unique/deterministic: file + what it is + where it is.
        return f"{file_path}::{qualified_name}::{chunk_type}:{start_line}-{end_line}"

    # -- imports -----------------------------------------------------

    @staticmethod
    def _extract_imports(root: Node, source_bytes: bytes) -> List[str]:
        imports = []
        for child in root.children:
            if child.type in ("import_statement", "import_from_statement"):
                imports.append(_node_text(child, source_bytes).strip())
        return imports
