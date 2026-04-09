# ==================================================
# ФИНАЛЬНЫЙ РАБОЧИЙ КОД: SHEAF LAPLACIAN + ДИФФУЗИЯ
# ДЛЯ СТРУКТУРИРОВАННЫХ ВХОДНЫХ ДАННЫХ (КВАДРАТНЫЕ СКОБКИ)
# ==================================================
!pip install -q scikit-learn numpy scipy

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# 1. ТОПОЛОГИЧЕСКИЕ МАТРИЦЫ
# --------------------------------------------------
M_equiv = np.eye(2)
M_implies = np.eye(2)
M_contra = -np.eye(2)

RELATION_MAP = {
    "equivalent": M_equiv,
    "implies": M_implies,
    "contradicts": M_contra
}

# --------------------------------------------------
# 2. SHEAF LAPLACIAN С ДИФФУЗИЕЙ
# --------------------------------------------------
class SparseSheafVerifier:
    def __init__(self, num_nodes, edges, edge_maps, feature_dim=2):
        self.N = num_nodes
        self.d = feature_dim
        self.L = sp.lil_matrix((self.N * self.d, self.N * self.d), dtype=np.float64)
        for (u, v), M_uv in edge_maps.items():
            idx_u = slice(u * self.d, (u + 1) * self.d)
            idx_v = slice(v * self.d, (v + 1) * self.d)
            self.L[idx_u, idx_u] += np.eye(self.d)
            self.L[idx_v, idx_v] += np.eye(self.d)
            self.L[idx_u, idx_v] -= M_uv.T
            self.L[idx_v, idx_u] -= M_uv
        self.L = self.L.tocsr()

    def compute_energy(self, state_vector):
        Lx = self.L.dot(state_vector)
        return float(np.linalg.norm(Lx) ** 2)

    def diffuse(self, state_vector, steps=50, lr=0.1):
        x = state_vector.copy()
        energies = [self.compute_energy(x)]
        for _ in range(steps):
            grad = self.L.dot(x)
            x = x - lr * grad
            energies.append(self.compute_energy(x))
        return x, energies

# --------------------------------------------------
# 3. ПАРСЕР СТРУКТУРИРОВАННОГО ТЕКСТА
# --------------------------------------------------
def parse_structured_text(text):
    """Извлекает триплеты из текста вида [A] implies [B]"""
    import re
    pattern = r'\[(.*?)\]\s+(implies|contradicts|equivalent)\s+\[(.*?)\]'
    matches = re.findall(pattern, text)
    triplets = [[sub, rel, obj] for sub, rel, obj in matches]
    return triplets

# --------------------------------------------------
# 4. ПОСТРОЕНИЕ ГРАФА И СОСТОЯНИЙ
# --------------------------------------------------
def build_sheaf_from_triplets(triplets):
    if not triplets:
        return None, None, None
    nodes = {}
    for subj, rel, obj in triplets:
        if subj not in nodes: nodes[subj] = len(nodes)
        if obj not in nodes: nodes[obj] = len(nodes)
    edge_maps = {}
    for subj, rel, obj in triplets:
        u, v = nodes[subj], nodes[obj]
        mat = RELATION_MAP.get(rel, M_equiv)
        edge_maps[(u, v)] = mat
    verifier = SparseSheafVerifier(len(nodes), list(edge_maps.keys()), edge_maps, feature_dim=2)
    state = np.zeros(len(nodes) * 2)
    first_node = sorted(nodes.keys())[0]
    state[nodes[first_node]*2 : nodes[first_node]*2+2] = [1.0, 0.0]
    changed = True
    for _ in range(100):
        changed = False
        for (u, v), M in edge_maps.items():
            idx_u = slice(u*2, u*2+2)
            idx_v = slice(v*2, v*2+2)
            if np.linalg.norm(state[idx_u]) > 1e-6 and np.linalg.norm(state[idx_v]) == 0:
                state[idx_v] = M @ state[idx_u]
                changed = True
            if np.linalg.norm(state[idx_v]) > 1e-6 and np.linalg.norm(state[idx_u]) == 0:
                state[idx_u] = M.T @ state[idx_v]
                changed = True
        if not changed:
            break
    return verifier, state, nodes

def get_node_state(nodes, state, node_name):
    if node_name not in nodes:
        return None
    idx = nodes[node_name]
    return state[idx*2 : idx*2+2]

# --------------------------------------------------
# 5. ДЕТЕКЦИЯ И ИСПРАВЛЕНИЕ
# --------------------------------------------------
def detect_and_correct(text, ground_truth=None):
    triplets = parse_structured_text(text)
    if len(triplets) < 2:
        return None, None, None
    premises = triplets[:-1]
    conclusion = triplets[-1]
    verifier, state, nodes = build_sheaf_from_triplets(premises)
    if verifier is None:
        return None, None, None
    subj, rel, obj = conclusion
    state_subj = get_node_state(nodes, state, subj)
    state_obj = get_node_state(nodes, state, obj)
    if state_subj is None or state_obj is None:
        return None, None, None
    expected_obj = RELATION_MAP.get(rel, M_equiv) @ state_subj
    energy_before = np.linalg.norm(expected_obj - state_obj) ** 2
    predicted = 1 if energy_before > 1e-5 else 0
    if predicted == 1:
        state_corrected = state.copy()
        idx_obj = nodes[obj]
        state_corrected[idx_obj*2 : idx_obj*2+2] = expected_obj
        _, energies = verifier.diffuse(state_corrected, steps=50, lr=0.1)
        energy_after = energies[-1]
    else:
        energy_after = energy_before
    return predicted, energy_before, energy_after

# --------------------------------------------------
# 6. ТЕСТОВЫЙ НАБОР (СТРУКТУРИРОВАННЫЙ)
# --------------------------------------------------
def create_dataset():
    return [
        ("[A] implies [B]. [B] implies [C]. Therefore [A] implies [C].", 0),
        ("[X] implies [Y]. [Y] contradicts [Z]. Therefore [X] contradicts [Z].", 0),
        ("[P] equivalent [Q]. [Q] equivalent [R]. So [P] equivalent [R].", 0),
        ("[A] implies [B]. [B] implies [C]. Therefore [A] contradicts [C].", 1),
        ("[X] implies [Y]. [Y] implies [Z]. Therefore [X] contradicts [Z].", 1),
        ("[P] contradicts [Q]. [Q] contradicts [R]. Therefore [P] equivalent [R].", 0),
        ("[Rain] implies [WetGround]. [WetGround] implies [GrassGrows]. Hence [Rain] implies [GrassGrows].", 0),
        ("[Rain] implies [WetGround]. [WetGround] implies [GrassGrows]. Therefore [Rain] contradicts [GrassGrows].", 1),
    ]

# --------------------------------------------------
# 7. ЗАПУСК
# --------------------------------------------------
def run_benchmark():
    dataset = create_dataset()
    y_true, y_pred, energies_before, energies_after = [], [], [], []
    valid = 0
    print("=== ДЕТЕКТОР + ИСПРАВИТЕЛЬ ГАЛЛЮЦИНАЦИЙ (SHEAF LAPLACIAN + ДИФФУЗИЯ) ===\n")
    for i, (text, label) in enumerate(dataset):
        res = detect_and_correct(text, label)
        if res[0] is None:
            print(f"{i+1}. ⚠️ ОШИБКА ПАРСИНГА | Truth={label}")
            continue
        pred, eng_before, eng_after = res
        y_true.append(label); y_pred.append(pred)
        energies_before.append(eng_before); energies_after.append(eng_after)
        valid += 1
        status = "✅" if pred == label else "❌"
        print(f"{i+1}. {status} | Truth={label} Pred={pred} | Energy before={eng_before:.4f} after={eng_after:.4f}")
        print(f"   {text[:70]}...\n")
    if valid == 0:
        print("Нет данных для оценки.")
        return
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== ИТОГОВАЯ ТОЧНОСТЬ НА {valid} ПРИМЕРАХ: {acc:.2%} ===")
    print(f"Средняя энергия галлюцинаций до: {np.mean([e for i,e in enumerate(energies_before) if y_true[i]==1]):.4f}")
    print(f"Средняя энергия галлюцинаций после: {np.mean([e for i,e in enumerate(energies_after) if y_true[i]==1]):.4f}")

if __name__ == "__main__":
    run_benchmark()
