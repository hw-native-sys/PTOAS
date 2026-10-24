#!/usr/bin/env python3
"""Expand the pressure fixture across zero/one/multiple K trips.

Check hardware flag balance plus every physical Mat/Left conflict and input's
latest writer. This is an event ancestry proof, without operation durations.
"""
import re
import sys


def require(value, message):
    if not value:
        raise AssertionError(message)


def loop_parts(lines):
    first = next(i for i, line in enumerate(lines) if line.strip().startswith('for (int64_t '))
    depth = 0
    for last in range(first, len(lines)):
        depth += lines[last].count('{') - lines[last].count('}')
        if depth == 0:
            return lines[:first], lines[first + 1:last], lines[last + 1:]
    raise AssertionError('unclosed loop')


def extract_function(text, name):
    begin = text.index('AICORE void ' + name + '(')
    brace = text.index('{', begin)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[brace + 1:end - 1].splitlines()


class Graph:
    def __init__(self, addresses, banks, sizes):
        self.addresses, self.banks, self.sizes = addresses, banks, sizes
        self.last, self.flags, self.ancestors, self.memory, self.versions = {}, {}, [], [], {}
        self.loads = self.copies = 0

    def op(self, pipe, dependencies=()):
        parents = set(dependencies)
        if pipe in self.last:
            parents.add(self.last[pipe])
        node = len(self.ancestors)
        bits = 0
        for parent in parents:
            bits |= self.ancestors[parent] | (1 << parent)
        self.ancestors.append(bits)
        self.last[pipe] = node
        return node

    def access(self, var, node, write):
        require(var in self.addresses, 'unresolved address ' + var)
        begin, end = self.addresses[var], self.addresses[var] + self.sizes[var]
        overlaps = [x for x in self.memory if x[0] == self.banks[var] and max(x[1], begin) < min(x[2], end)]
        for old in overlaps:
            if write or old[4]:
                require(self.ancestors[node] & (1 << old[3]), 'unordered physical conflict for ' + var)
        if write:
            self.versions[var] = node
        else:
            latest = next((x for x in reversed(overlaps) if x[4]), None)
            require(latest is not None and latest[5] == var and latest[3] == self.versions[var], 'stale source ' + var)
        self.memory.append((self.banks[var], begin, end, node, write, var))

    def segment(self, lines):
        for raw in lines:
            line = raw.strip()
            event = re.fullmatch(r'(set_flag|wait_flag)\(PIPE_(\w+), PIPE_(\w+), EVENT_ID(\d+)\);', line)
            load = re.fullmatch(r'TLOAD\((v\d+), (v\d+)\);', line)
            copy = re.fullmatch(r'TEXTRACT\((v\d+), (v\d+), (v\d+), (v\d+)\);', line)
            barrier = re.fullmatch(r'pipe_barrier\(PIPE_(\w+)\);', line)
            if event:
                kind, source, target, number = event.groups()
                key = source, target, int(number)
                require(0 <= int(number) < 8, 'illegal hardware event ID')
                if kind == 'set_flag':
                    require(key not in self.flags, 'set over outstanding flag ' + str(key))
                    self.flags[key] = self.op(source)
                else:
                    require(key in self.flags, 'wait without matching set ' + str(key))
                    self.op(target, [self.flags.pop(key)])
            elif barrier:
                if barrier[1] == 'ALL':
                    deps = list(self.last.values())
                    for pipe in ('MTE2', 'MTE1', 'M', 'FIX', 'S'):
                        self.op(pipe, deps)
                else:
                    self.op(barrier[1])
            elif load:
                self.access(load[1], self.op('MTE2'), True)
                self.loads += 1
            elif copy:
                node = self.op('MTE1')
                self.access(copy[2], node, False)
                self.access(copy[1], node, True)
                self.copies += 1
            elif line.startswith(('TLOAD', 'TEXTRACT', 'set_flag', 'wait_flag', 'pipe_barrier')):
                raise AssertionError('unknown operation ' + line)


def audit(lines, trips, dynamic_arg=None, dynamic_value=65536):
    values, addresses, banks, sizes = {}, {}, {}, {}
    if dynamic_arg is not None:
        # A concrete legal, nonoverlapping runtime instantiation; this does
        # not establish correctness for arbitrary runtime aliasing.
        values[dynamic_arg] = dynamic_value
    for line in lines:
        match = re.search(r'const int(?:32|64)_t (v\d+) = (-?\d+);', line)
        if match:
            values[match[1]] = int(match[2])
        match = re.search(r'(v\d+) = \(\w+\) (v\d+);', line)
        if match and match[2] in values:
            values[match[1]] = values[match[2]]
        match = re.search(r'Tile<TileType::(Mat|Left), half, (\d+), (\d+), [^;]+?> (v\d+)(?: =|;)', line)
        if match:
            banks[match[4]] = match[1]
            sizes[match[4]] = int(match[2]) * int(match[3]) * 2
        match = re.search(r'TASSIGN\((v\d+), (v\d+)\);', line)
        if match and match[2] in values:
            addresses[match[1]] = values[match[2]]
    prologue, output, epilogue = loop_parts(lines)
    head, inner, tail = loop_parts(output)
    graph = Graph(addresses, banks, sizes)
    graph.segment(prologue)
    for _ in range(3):
        graph.segment(head)
        for _ in range(trips):
            graph.segment(inner)
        graph.segment(tail)
    graph.segment(epilogue)
    require(not graph.flags, 'undrained events after repeated outputs: ' + str(graph.flags))
    require(graph.loads == 3 * (3 + 6 * trips), 'unexpected TLOAD count')
    require(graph.copies == 3 * (3 + 12 * trips), 'unexpected TEXTRACT count')
    return graph


text = open(sys.argv[1]).read()
mode = sys.argv[2] if len(sys.argv) > 2 else 'positive'
require(mode in ('positive', 'dynamic', 'manual'), 'unknown fixture mode')
name = {'positive': 'coalesce_pressure', 'dynamic': 'keep_unknown_mat_address', 'manual': 'keep_explicit_event_id'}[mode]
lines = extract_function(text, name)
if mode == 'manual':
    events = re.findall(r'(set_flag|wait_flag)\(PIPE_(\w+), PIPE_(\w+), EVENT_ID(\d+)\);', '\n'.join(lines))
    require(events == [('set_flag', 'M', 'MTE1', '7'), ('wait_flag', 'M', 'MTE1', '7')], 'manual function was unexpectedly auto-synchronized')
    require(not any('pipe_barrier(PIPE_ALL)' in x for x in lines), 'unexpected automatic barrier in skipped manual function')
    print('Manual fixture: existing whole-function skip preserved; separate G1 failure is required')
    sys.exit(0)
dynamic_arg = None
if mode == 'dynamic':
    signature = re.search(r'AICORE void ' + name + r'\(([^)]*)\)', text)
    args = re.findall(r'\bv\d+\b', signature[1])
    require(len(args) == 4, 'unexpected dynamic fixture signature')
    dynamic_arg = args[-1]
else:
    require(not any('pipe_barrier(PIPE_ALL)' in x for x in lines), 'pressure still falls back to a full barrier')
if mode == 'positive':
    for negative in ('keep_branch_lifetime', 'keep_nested_lifetime'):
        body = '\n'.join(extract_function(text, negative))
        early = body.index('// pto: %v1_4')
        late = body.index('// pto: %v1_5')
        release_pattern = r'set_flag\(PIPE_MTE1, PIPE_MTE2, EVENT_ID(\d+)\);'
        before = re.findall(release_pattern, body[early:late])
        after = re.findall(release_pattern, body[late:])
        require(len(before) == 1 and len(after) == 1 and before != after,
                'control-flow scope lost its separate releases: ' + negative)
for trips in (0, 1, 3, 9):
    audit(lines, trips, dynamic_arg)
if mode == 'dynamic':
    rejected = False
    try:
        audit(lines, 3, dynamic_arg, dynamic_value=24576)
    except AssertionError:
        rejected = True
    require(rejected, 'invalid dynamic alias that overwrites a live input was not rejected')
    print('Dynamic fixture: concrete dynamic=65536, 0/1/3/9 K trips and repeated outputs passed physical event proof; invalid alias rejected; G1 diagnostic remains separately required')
    sys.exit(0)
# Removing only the shared release must fail on multi-trip reuse; a checker
# that accepts all graphs cannot establish the optimization's correctness.
removed = False
prologue, output, epilogue = loop_parts(lines)
head, inner, tail = loop_parts(output)
last_copy = max(i for i, x in enumerate(inner) if x.strip().startswith('TEXTRACT('))
release = next(i for i in range(last_copy + 1, len(inner)) if 'set_flag(PIPE_MTE1, PIPE_MTE2,' in inner[i])
inner = inner[:release] + inner[release + 1:]
try:
    values_graph = audit(lines, 3)
    broken = Graph(values_graph.addresses, values_graph.banks, values_graph.sizes)
    broken.segment(prologue)
    broken.segment(head)
    for _ in range(3):
        broken.segment(inner)
except AssertionError:
    removed = True
require(removed, 'missing shared release negative control was not rejected')
print('Mat backedge coalescing: 0/1/3/9 K trips and repeated outputs passed; missing-release control rejected')
