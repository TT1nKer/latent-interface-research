"""Read the LEAPS Karel representation and validate native program execution."""
from karel_predicates import observe
from karel_return_pairs import decisions
from karel_trace_control import DIRECTIONS, World

ACTIONS = ('move', 'turnLeft', 'turnRight', 'pickMarker', 'putMarker')
PREDICATES = ('frontIsClear', 'leftIsClear', 'rightIsClear', 'markersPresent', 'noMarkersPresent')
VOCAB = ('DEF', 'run', 'm(', 'm)', 'move', 'turnRight', 'turnLeft', 'pickMarker', 'putMarker',
         'r(', 'r)', *(f'R={i}' for i in range(20)), 'REPEAT', 'c(', 'c)', 'i(', 'i)', 'e(',
         'e)', 'IF', 'IFELSE', 'ELSE', *PREDICATES, 'not', 'w(', 'w)', 'WHILE')


class LeapsWorld(World):
    def step(self, action):
        if action == 'putMarker' and self.markers.get(self.hero, 0) >= 9:
            return False
        return super().step(action)


def parse_program(tokens):
    stream = iter(tokens)

    def take(expected):
        actual = next(stream, None)
        if actual != expected:
            raise ValueError(f'Expected {expected!r}, received {actual!r}')

    def condition():
        take('c(')
        name = next(stream)
        if name == 'not':
            node = {'type': 'not', 'condition': condition()}
        elif name in PREDICATES:
            node = {'type': name}
        else:
            raise ValueError(f'Unknown condition {name!r}')
        take('c)')
        return node

    def block(letter):
        take(letter + '(')
        nodes = []
        for name in stream:
            if name == letter + ')':
                return nodes
            if name in ACTIONS:
                node = {'type': name}
            elif name == 'REPEAT':
                count = next(stream)
                if not count.startswith('R='):
                    raise ValueError(f'Invalid repeat count {count!r}')
                node = {'type': 'repeat', 'times': int(count[2:]), 'body': block('r')}
            elif name in ('WHILE', 'IF'):
                node = {'type': name.lower(), 'condition': condition(),
                        'body': block('w' if name == 'WHILE' else 'i')}
            elif name == 'IFELSE':
                node = {'type': 'ifElse', 'condition': condition(), 'ifBody': block('i')}
                take('ELSE')
                node['elseBody'] = block('e')
            else:
                raise ValueError(f'Unknown statement {name!r}')
            nodes.append(node)
        raise ValueError('Missing closing bracket')

    take('DEF')
    take('run')
    result = {'run': block('m')}
    take(None)
    return result


def decode_state(array):
    import numpy as np
    if array.ndim != 3 or array.shape[2] != 16 or array.dtype != np.bool_:
        raise ValueError(f'Unexpected state type/shape {array.dtype}/{array.shape}')
    rows, cols, _ = array.shape
    if not (array[:, :, 5:].sum(axis=2) == 1).all():
        raise ValueError('Wrong number of per-cell marker planes')
    poses = np.argwhere(array[:, :, :4])
    if len(poses) != 1:
        raise ValueError('World requires one pose and heading')
    r, c, direction = map(int, poses[0])
    blocked = {(rows-1-int(i), int(j)) for i, j in np.argwhere(array[:, :, 4])}
    markers = {(rows-1-int(i), int(j)):int(k)+1 for i, j, k in np.argwhere(array[:, :, 6:])}
    hero = rows-1-r, c
    if hero in blocked or set(markers) & blocked:
        raise ValueError('Robot or marker inside a blocked cell')
    return LeapsWorld(rows, cols, blocked, markers, hero, direction)


def check_sensors(array, world):
    import numpy as np
    [(r,c,d)] = np.argwhere(array[:, :, :4])
    native_deltas = ((-1,0),(0,1),(1,0),(0,-1))
    bits = []
    for turn in (0,-1,1):
        dr,dc = native_deltas[(int(d)+turn)%4]
        i,j = int(r)+dr,int(c)+dc
        bits.append(0 <= i < array.shape[0] and 0 <= j < array.shape[1] and not array[i,j,4])
    bits.append(bool(array[r,c,6:].any()))
    if tuple(bits) != observe(world):
        raise ValueError('Native y-down and normalized four-predicate observations differ')


def normalized_grid(world):
    r, c = world.hero
    return {'rows': world.rows, 'cols': world.cols, 'crashed': False,
            'hero': f'{r}:{c}:{DIRECTIONS[world.direction]}',
            'blocked': ' '.join(f'{i}:{j}' for i,j in sorted(world.blocked)),
            'markers': ' '.join(f'{i}:{j}:{n}' for (i,j),n in sorted(world.markers.items()))}


def convert(group, index, guid, counts):
    program = [VOCAB[int(t)] for t in group['program'][:]]
    if len(program) != int(guid.split('_')[4]):
        raise ValueError(f'Incorrect program length in {guid}')
    ast = parse_program(program)
    states, acts = group['s_h'][:6], group['a_h'][:6]
    state_sizes, action_sizes = group['s_h_len'][:], group['a_h_len'][:]
    if len(state_sizes) != 10 or len(action_sizes) != 10:
        raise ValueError(f'Expected 10 demonstrations in {guid}')
    examples = []
    for j in range(6):
        ns, na = int(state_sizes[j]), int(action_sizes[j])
        if not (ns == na+1 and 1 <= ns <= states.shape[1] and 0 <= na <= acts.shape[1]):
            raise ValueError(f'Invalid padding lengths in {guid}, world {j}: {ns}/{na}')
        world = decode_state(states[j,0]); before = normalized_grid(world)
        check_sensors(states[j,0],world)
        if world.state() != LeapsWorld.parse(before).state():
            raise ValueError('Normalization changed the starting world')
        cursor = decisions(ast, world)
        actions = []
        for k in range(na):
            expected = ACTIONS[int(acts[j,k])]
            actual = next(cursor, None)
            if actual != expected or not world.step(expected):
                raise ValueError(f'{guid}/{j}/{k} expected {expected}, interpreted {actual}')
            if world.state() != decode_state(states[j,k+1]).state():
                raise ValueError(f'{guid}/{j}/{k} native step differs from stored state')
            check_sensors(states[j,k+1],world)
            actions.append(actual)
        if next(cursor, None) is not None:
            raise ValueError(f'{guid}/{j} program did not terminate at recorded time')
        after = normalized_grid(world)
        if LeapsWorld.parse(after).state() != world.state():
            raise ValueError('Normalization changed the endpoint')
        examples.append({'example_index':j, 'inpgrid_json':before, 'outgrid_json':after, 'actions':actions})
        counts.update({'selected_worlds':1, 'validated_stop':1, 'native_states':ns,
                       'native_steps':na, 'native_observations':ns, 'zero_action_worlds':int(not actions),
                       'no_move_worlds':int('move' not in actions)})
    return {'guid':guid, 'source_index':index, 'program_tokens':program,
            'program_json':ast, 'examples':examples}
