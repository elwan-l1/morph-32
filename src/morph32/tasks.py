"""Fixed reasoning and programming smoke-test prompts."""

TASKS = [
    (
        "reason_arithmetic",
        "A store has 17 boxes with 24 bolts each. It sells 137 bolts and then receives 5 boxes with 18 bolts each. How many bolts remain? Answer with only the integer.",
        "361",
    ),
    (
        "reason_logic",
        "Exactly one of A, B, C is true. A says B is false. B says C is false. C says both A and B are false. Which statement is true? Answer A, B, or C.",
        "B",
    ),
    (
        "reason_probability",
        "A fair six-sided die is rolled twice. Given that at least one roll is 6, what is the probability that both rolls are 6? Give only the fraction.",
        "1/11",
    ),
    (
        "reason_schedule",
        "A task starts at 09:35, takes 2 hours 48 minutes, pauses for 37 minutes, then runs 1 hour 26 minutes. What time does it finish? Answer HH:MM only.",
        "14:26",
    ),
    (
        "coding_intervals",
        "Write Python function merge_intervals(intervals) that returns sorted merged closed intervals, merging touching endpoints. Empty input returns []. Return only one Python code block. No imports.",
        None,
    ),
    (
        "coding_brackets",
        "Write Python function balanced(text) that checks matching (), [], {} brackets, ignoring all other characters. Return only one Python code block. No imports.",
        None,
    ),
    (
        "coding_search",
        "Write Python function lower_bound(values, target) for a sorted list. Return first index whose value is >= target, or len(values). Use O(log n) time. Return only one Python code block. No imports.",
        None,
    ),
    (
        "coding_unique",
        "Write Python function first_unique(text) that returns the first character appearing exactly once, or None. Use O(n) time. Return only one Python code block. No imports.",
        None,
    ),
]
