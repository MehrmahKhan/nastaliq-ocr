"""Character and word error rate.

Implemented by hand with explicit loops so the pipeline does not depend on a
metric library behaving a certain way, and so the numbers are easy to defend
in a viva.
"""


def edit_distance(reference, hypothesis):
    """Classic Levenshtein distance over two sequences (strings or lists)."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1

    table = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(0)
        table.append(row)

    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1

            deletion = table[i - 1][j] + 1
            insertion = table[i][j - 1] + 1
            substitution = table[i - 1][j - 1] + cost

            best = deletion
            if insertion < best:
                best = insertion
            if substitution < best:
                best = substitution

            table[i][j] = best

    return table[rows - 1][cols - 1]


def character_error_rate(references, hypotheses):
    total_distance = 0
    total_length = 0

    for index in range(len(references)):
        reference = references[index]
        hypothesis = hypotheses[index]
        total_distance = total_distance + edit_distance(reference, hypothesis)
        total_length = total_length + len(reference)

    if total_length == 0:
        return 0.0
    return float(total_distance) / float(total_length)


def word_error_rate(references, hypotheses):
    total_distance = 0
    total_length = 0

    for index in range(len(references)):
        reference_words = references[index].split()
        hypothesis_words = hypotheses[index].split()
        total_distance = total_distance + edit_distance(reference_words, hypothesis_words)
        total_length = total_length + len(reference_words)

    if total_length == 0:
        return 0.0
    return float(total_distance) / float(total_length)


def exact_match_rate(references, hypotheses):
    matched = 0
    for index in range(len(references)):
        if references[index] == hypotheses[index]:
            matched = matched + 1
    if len(references) == 0:
        return 0.0
    return float(matched) / float(len(references))
