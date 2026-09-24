"""Equivalent compact-journal replay with a bounded cumulative-text cache.

This optional reader does not modify journals, writers, or frozen comparison
code. Pass its decoded iterable to ``canonical_outcomes(..., journal=...)``.
Definitions remain retained as in the reference reader; only reconstructed
text states use the LRU cache. Cache eviction affects speed, never evidence.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from pdblend.results.journal import SCHEMA, iter_jsonl


@dataclass
class JournalReadStats:
    rows: int = 0
    payloads: int = 0
    patch_definitions_visited: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_peak_entries: int = 0
    cache_peak_characters: int = 0


def iter_comparison_journal(path, *, max_cache_entries=512,
                            max_cache_characters=8_388_608, stats=None):
    """Yield exactly the reference reader's rows, preserving strict failures.

    Limits bound retained reconstructed states, in entries and Unicode code
    points (field names plus text, at least one per field; not bytes). A state larger than the budget is never
    retained. Set either limit to zero for uncached replay. Forward/missing
    references, cycles, duplicate definitions, invalid patches and truncated
    gzip remain errors; there is no recovery or event filtering.
    """
    if any(type(value) is not int or value < 0
           for value in (max_cache_entries, max_cache_characters)):
        raise ValueError('cache limits must be nonnegative integers')
    stats = JournalReadStats() if stats is None else stats
    definitions, cache = {}, OrderedDict()
    cache_characters = 0
    for row in iter_jsonl(path):
        stats.rows += 1
        schema = row.pop('_journal_schema', None)
        if schema is None:
            yield row
            continue
        if schema != SCHEMA:
            raise ValueError('unsupported compact journal schema: '+str(schema))
        if 'payload_ref' in row:
            stats.payloads += 1
            reference = row.pop('payload_ref')
            definition = row.pop('payload_data', None)
            if definition is not None:
                if reference in definitions:
                    raise ValueError('duplicate payload definition')
                definitions[reference] = definition
            if reference not in definitions:
                raise ValueError('unresolved compact payload reference')
            chain, cursor, seen = [], reference, set()
            while cursor is not None and cursor not in cache:
                if cursor in seen or cursor not in definitions:
                    raise ValueError('invalid compact payload text chain')
                seen.add(cursor)
                item = definitions[cursor]
                chain.append(item)
                cursor = item['previous']
            if cursor is None:
                stats.cache_misses += 1
                texts = {}
            else:
                # Every cached state was validated through a terminated chain.
                # Definitions are immutable once inserted, so this prefix
                # cannot subsequently become missing or cyclic.
                stats.cache_hits += 1
                cache.move_to_end(cursor)
                texts = dict(cache[cursor][0])
            for item in reversed(chain):
                stats.patch_definitions_visited += 1
                for key, (prefix, suffix) in item['text_patches'].items():
                    prior = texts.get(key, '')
                    if type(prefix) is not int or not 0 <= prefix <= len(prior):
                        raise ValueError('invalid compact text prefix')
                    texts[key] = prior[:prefix]+suffix
            body = dict(definitions[reference]['body'])
            if 'text' in definitions[reference]['text_patches']:
                body['text'] = texts['text']
            if isinstance(body.get('choices'), list):
                choices = []
                for index, choice in enumerate(body['choices']):
                    choice = dict(choice)
                    if choice.pop('_text_from_payload', False):
                        choice['text'] = body['text']
                    elif f'choice:{index}' in definitions[reference]['text_patches']:
                        choice['text'] = texts[f'choice:{index}']
                    choices.append(choice)
                body['choices'] = choices
            row['payload'] = body
            size = sum(max(1, len(key)) + len(text) for key, text in texts.items())
            if max_cache_entries and size <= max_cache_characters and max_cache_characters:
                if reference in cache:
                    cache_characters -= cache.pop(reference)[1]
                while cache and (len(cache) >= max_cache_entries
                                 or cache_characters + size > max_cache_characters):
                    _, (_, removed_size) = cache.popitem(last=False)
                    cache_characters -= removed_size
                cache[reference] = (texts, size)
                cache_characters += size
                stats.cache_peak_entries = max(stats.cache_peak_entries, len(cache))
                stats.cache_peak_characters = max(stats.cache_peak_characters, cache_characters)
        if 'token_payload_ref' in row:
            reference = row.pop('token_payload_ref')
            if reference not in definitions:
                raise ValueError('unresolved compact token reference')
            row['token_ids'] = list(definitions[reference]['body']['token_ids'])
        yield row
