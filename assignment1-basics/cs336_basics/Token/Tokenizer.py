import regex as re
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries


class BPETrainer_V1:  # version 1.0, not fast enough
    def __init__(self):
        self.num_processes = 4
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def string_to_bytes(self, s: str) -> list[bytes]:
        return [bytes([c]) for c in s.encode('utf-8')]

    def init_vocab(self, special_tokens: list) -> dict:
        vocab = {i: bytes([i]) for i in range(256)}
        for token in special_tokens:
            vocab[len(vocab)] = token.encode('utf-8') if isinstance(token, str) else token
        return vocab

    def count_word(self, input_path: str, special_tokens: list, start, end) -> dict[tuple[bytes], int]:
        word_count = defaultdict(int)
        special_tokens_joined = "|".join(re.escape(token) for token in special_tokens)
        with open(input_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors="ignore")
            for part in re.split(f"({special_tokens_joined})", chunk):
                if part in special_tokens:
                    continue
                for matched in re.finditer(self.PAT, part):
                    token = tuple(self.string_to_bytes(matched.group(0)))
                    word_count[token] += 1
        return word_count

    """ [parallel failed]
    def pair_count(self, word_count: dict) -> dict[tuple[bytes, bytes], int]:
        pair_count = defaultdict(int)
        for token, count in word_count.items():
            for i in range(len(token) - 1):
                pair_count[(token[i], token[i + 1])] += count
        return pair_count

    def pair_merge(self, pair_best, word_count: dict) -> dict[tuple[bytes, bytes], int]:
        word_count_new = defaultdict(int)
        for token, count in word_count.items():
            temp_list = []
            i = 0
            while i < len(token):
                if i + 1 != len(token) and (token[i], token[i + 1]) == pair_best:
                    temp_list.append(token[i] + token[i + 1])
                    i += 2
                else:
                    temp_list.append(token[i])
                    i += 1
            token_new = tuple(temp_list)
            word_count_new[token_new] += count
        return word_count_new
    """

    def merge(self, vocab_size: int, vocab: dict, word_count : dict) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # Loop merge
        merged = []
        
        while len(vocab) < vocab_size:
            # Initialize the count of pair
            pair_count = defaultdict(int)
            for token, count in word_count.items():
                for i in range(len(token) - 1):
                    pair_count[(token[i], token[i + 1])] += count
            """ [parallel failed]
            it = iter(word_count.items())
            word_count_splited = [
                dict(islice(it, len(word_count) // self.num_processes + (i < len(word_count) % self.num_processes))) 
                for i in range(self.num_processes)]
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                results = pool.map(self.pair_count, word_count_splited) 
            for result in results:
                for pair, count in result.items():
                    pair_count[pair] += count
            """

            # Find the most frequent pair with greast lexicographic order
            count_max = max(pair_count.values())
            pair_best = max([pair for pair, count in pair_count.items() if count == count_max])

            # If there exists pair to merge, then do it
            vocab[len(vocab)] = pair_best[0] + pair_best[1]
            merged.append(pair_best)
            word_count_new = defaultdict(int)
            for token, count in word_count.items():
                temp_list = []
                i = 0
                while i < len(token):
                    if i + 1 != len(token) and (token[i], token[i + 1]) == pair_best:
                        temp_list.append(token[i] + token[i + 1])
                        i += 2
                    else:
                        temp_list.append(token[i])
                        i += 1
                token_new = tuple(temp_list)
                word_count_new[token_new] += count
            word_count = word_count_new

        return vocab, merged

    def train_BPE(self, input_path: str, vocab_size: int, special_tokens: list) -> tuple[dict, list]:
        # Initialization
        vocab = self.init_vocab(special_tokens)

        # Find boundaries and split strings into trunks
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")

        # Count words in every trunk
        word_count = defaultdict(int)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            result = self.count_word(input_path, special_tokens, start, end)
            for token, count in result.items():
                word_count[token] += count
        """ [parallel failed]
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = pool.starmap(self.count_word, [(input_path, special_tokens, start, end) 
                                                     for start, end in zip(boundaries[:-1], boundaries[1:])])
        for result in results:
            for token, count in result.items():
                word_count[token] += count
        """

        # Merge
        vocab, merged = self.merge(vocab_size, vocab, word_count)

        return vocab, merged


class BPETrainer:
    def __init__(self):
        self.num_processes = 4
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def string_to_bytes(self, s: str) -> list[bytes]:
        return [bytes([c]) for c in s.encode('utf-8')]

    def init_vocab(self, special_tokens: list) -> dict:
        vocab = {i: bytes([i]) for i in range(256)}
        for token in special_tokens:
            vocab[len(vocab)] = token.encode('utf-8') if isinstance(token, str) else token
        return vocab

    def count_word(self, input_path: str, special_tokens: list, start, end) -> dict[tuple[bytes], int]:
        word_count = defaultdict(int)
        special_tokens_joined = "|".join(re.escape(token) for token in special_tokens)
        with open(input_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors="ignore")
            for part in re.split(f"({special_tokens_joined})", chunk):
                if part in special_tokens:
                    continue
                for matched in re.finditer(self.PAT, part):
                    token = tuple(self.string_to_bytes(matched.group(0)))
                    word_count[token] += 1
        return word_count

    def initialize(self, word_count: dict) -> tuple[dict, dict]:
        # Initialize the count of pair
        pair_count = defaultdict(int)
        pair2word = defaultdict(lambda : defaultdict(int))
        for word, count in word_count.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_count[pair] += count
                pair2word[pair][word] += 1
        return pair_count, pair2word

    def merge(self, vocab_size: int, vocab: dict, word_count : dict) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # Loop merge
        merged = []
        pair_count, pair2word = self.initialize(word_count)
        while len(vocab) < vocab_size:
            # Find the most frequent pair with greast lexicographic order
            count_max = max(pair_count.values())
            pair_best = max([pair for pair, count in pair_count.items() if count == count_max])

            # If there exists pair to merge, then do it
            vocab[len(vocab)] = pair_best[0] + pair_best[1]
            merged.append(pair_best)
            
            # Update pair_count and others
            for word in list(pair2word[pair_best].keys()):
                count = word_count[word]
                if not count:
                    continue
                
                # Remove old word
                word_count[word] -= count
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_count[pair] -= count
                    pair2word[pair][word] -= 1
                    if pair_count[pair] == 0:
                        del pair_count[pair]
                    if pair2word[pair][word] == 0:
                        del pair2word[pair][word]

                # Get new word
                temp_list = []
                i = 0
                while i < len(word):
                    if i + 1 != len(word) and (word[i], word[i + 1]) == pair_best:
                        temp_list.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        temp_list.append(word[i])
                        i += 1
                word_new = tuple(temp_list)

                # Update new word
                word_count[word_new] += count
                for i in range(len(word_new) - 1):
                    pair = (word_new[i], word_new[i + 1])
                    pair_count[pair] += count
                    pair2word[pair][word_new] += 1
        return vocab, merged

    def train_BPE(self, input_path: str, vocab_size: int, special_tokens: list) -> tuple[dict, list]:
        # Initialization
        vocab = self.init_vocab(special_tokens)

        # Find boundaries and split strings into trunks
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")

        # Count words in every trunk
        word_count = defaultdict(int)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            result = self.count_word(input_path, special_tokens, start, end)
            for token, count in result.items():
                word_count[token] += count

        # Merge
        vocab, merged = self.merge(vocab_size, vocab, word_count)

        return vocab, merged
