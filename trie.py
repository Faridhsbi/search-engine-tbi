"""
trie.py

Trie (Prefix Tree) data structure for efficient dictionary storage
and prefix-based querying. Supports:
  - Exact term lookup
  - Prefix search (for autocomplete)
  - Wildcard search
  - Serialization/Deserialization

This is a more complex and efficient dictionary structure compared
to a simple Python dict, as required for bonus points.
"""

import pickle


class TrieNode:
    """
    A single node in the Trie.

    Attributes
    ----------
    children : dict
        Mapping from character to child TrieNode
    is_terminal : bool
        Whether this node represents the end of a valid term
    term_id : int or None
        The term ID associated with this term (if is_terminal)
    """
    __slots__ = ['children', 'is_terminal', 'term_id']

    def __init__(self):
        self.children = {}
        self.is_terminal = False
        self.term_id = None


class Trie:
    """
    A Trie (prefix tree) for storing term → term_id mappings.

    Supports efficient:
      - Insertion: O(|term|)
      - Exact lookup: O(|term|)
      - Prefix search: O(|prefix| + number of matches)
      - Wildcard search with '*' and '?'

    Methods
    -------
    insert(term, term_id)
        Insert a term with its associated term_id.
    search(term) -> Optional[int]
        Exact match lookup. Returns term_id or None.
    prefix_search(prefix) -> List[(str, int)]
        Return all (term, term_id) pairs that start with the given prefix.
    wildcard_search(pattern) -> List[(str, int)]
        Return all (term, term_id) pairs matching a wildcard pattern.
        Supports '*' (any sequence) and '?' (single char).
    get_all_terms() -> List[(str, int)]
        Return all (term, term_id) pairs stored in the Trie.
    """

    def __init__(self):
        self.root = TrieNode()
        self._size = 0

    def __len__(self):
        return self._size

    def insert(self, term, term_id):
        """
        Insert a term with its associated term_id into the Trie.

        Parameters
        ----------
        term : str
            The term to insert
        term_id : int
            The integer ID for this term
        """
        node = self.root
        for char in term:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if not node.is_terminal:
            self._size += 1
        node.is_terminal = True
        node.term_id = term_id

    def search(self, term):
        """
        Search for an exact term in the Trie.

        Parameters
        ----------
        term : str

        Returns
        -------
        int or None
            The term_id if found, None otherwise
        """
        node = self.root
        for char in term:
            if char not in node.children:
                return None
            node = node.children[char]
        if node.is_terminal:
            return node.term_id
        return None

    def prefix_search(self, prefix):
        """
        Find all terms that start with the given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        List[Tuple[str, int]]
            List of (term, term_id) pairs matching the prefix
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        results = []
        self._collect_all(node, prefix, results)
        return results

    def wildcard_search(self, pattern):
        """
        Find all terms matching a wildcard pattern.
        Supports:
          '*' — matches any sequence of characters (including empty)
          '?' — matches exactly one character

        Parameters
        ----------
        pattern : str

        Returns
        -------
        List[Tuple[str, int]]
        """
        results = []
        self._wildcard_helper(self.root, pattern, 0, "", results)
        return results

    def get_all_terms(self):
        """
        Return all (term, term_id) pairs in the Trie.

        Returns
        -------
        List[Tuple[str, int]]
        """
        results = []
        self._collect_all(self.root, "", results)
        return results

    def _collect_all(self, node, prefix, results):
        """Recursively collect all terminal nodes under the given node."""
        if node.is_terminal:
            results.append((prefix, node.term_id))
        for char in sorted(node.children.keys()):
            self._collect_all(node.children[char], prefix + char, results)

    def _wildcard_helper(self, node, pattern, idx, current, results):
        """Recursive helper for wildcard matching."""
        if idx == len(pattern):
            if node.is_terminal:
                results.append((current, node.term_id))
            return

        ch = pattern[idx]
        if ch == '?':
            # Match exactly one character
            for c, child in node.children.items():
                self._wildcard_helper(child, pattern, idx + 1, current + c, results)
        elif ch == '*':
            # Match zero characters (skip the *)
            self._wildcard_helper(node, pattern, idx + 1, current, results)
            # Match one or more characters
            for c, child in node.children.items():
                self._wildcard_helper(child, pattern, idx, current + c, results)
        else:
            # Exact character match
            if ch in node.children:
                self._wildcard_helper(node.children[ch], pattern, idx + 1, current + ch, results)

    def serialize(self):
        """
        Serialize the Trie to bytes using pickle.

        Returns
        -------
        bytes
        """
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data):
        """
        Deserialize a Trie from bytes.

        Parameters
        ----------
        data : bytes

        Returns
        -------
        Trie
        """
        return pickle.loads(data)

    def save(self, filepath):
        """Save the Trie to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load a Trie from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Test the Trie
    trie = Trie()

    terms = ["apple", "application", "apply", "banana", "band", "bandana", "cat"]
    for i, term in enumerate(terms):
        trie.insert(term, i)

    print(f"Trie size: {len(trie)}")

    # Exact search
    for term in terms:
        tid = trie.search(term)
        print(f"  search('{term}') = {tid}")

    assert trie.search("apple") == 0
    assert trie.search("xyz") is None

    # Prefix search
    print(f"\n  prefix_search('app') = {trie.prefix_search('app')}")
    print(f"  prefix_search('ban') = {trie.prefix_search('ban')}")

    # Wildcard search
    print(f"\n  wildcard_search('app*') = {trie.wildcard_search('app*')}")
    print(f"  wildcard_search('b?n*') = {trie.wildcard_search('b?n*')}")
    print(f"  wildcard_search('*ana') = {trie.wildcard_search('*ana')}")

    # Serialization roundtrip
    data = trie.serialize()
    trie2 = Trie.deserialize(data)
    assert trie2.search("apple") == 0
    assert len(trie2) == len(trie)
    print("\n[OK] All Trie tests PASSED")
