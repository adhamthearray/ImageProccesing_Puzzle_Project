import os
import cv2
import numpy as np
from collections import deque, defaultdict

# ==========================================================
# 0) LAPLACIAN EDGE ENHANCEMENT (COLOR PIECES)
# ==========================================================
def laplacian_enhance_bgr(img_bgr, alpha=0.8, ksize=3):
    smooth = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=ksize)
    lap = cv2.convertScaleAbs(lap)
    lap3 = cv2.merge([lap, lap, lap])
    enhanced = cv2.addWeighted(smooth, 1.0, lap3, alpha, 0)
    return enhanced

# ==========================================================
# 1) COLOR EDGE STRIPS
# ==========================================================
def extract_sides_color(img, border=5):
    top = img[0:border, :, :]
    bottom = img[-border:, :, :]
    left = img[:, 0:border, :]
    right = img[:, -border:, :]
    return top, right, bottom, left

# ==========================================================
# 2) HSV HIST DISTANCE (with flip option)
# ==========================================================
def edge_distance_color_hist(edge1_bgr, edge2_bgr):
    e1 = cv2.cvtColor(edge1_bgr, cv2.COLOR_BGR2HSV)
    e2 = cv2.cvtColor(edge2_bgr, cv2.COLOR_BGR2HSV)
    e2_flip = cv2.flip(e2, 0)

    def mask_background(edge_hsv):
        pixels = edge_hsv.reshape(-1, 3)
        counts = {}
        for px in map(tuple, pixels):
            counts[px] = counts.get(px, 0) + 1
        bg_color = max(counts, key=counts.get)
        mask = np.any(edge_hsv != bg_color, axis=2).astype(np.uint8)
        return mask

    m1  = mask_background(e1)
    m2  = mask_background(e2)
    m2f = mask_background(e2_flip)

    hist1  = cv2.calcHist([e1], [0,1,2], m1,  [8,8,8], [0,180,0,256,0,256])
    hist2  = cv2.calcHist([e2], [0,1,2], m2,  [8,8,8], [0,180,0,256,0,256])
    hist2f = cv2.calcHist([e2_flip], [0,1,2], m2f, [8,8,8], [0,180,0,256,0,256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    cv2.normalize(hist2f, hist2f)

    sim1 = cv2.compareHist(hist1, hist2,  cv2.HISTCMP_CORREL)
    sim2 = cv2.compareHist(hist1, hist2f, cv2.HISTCMP_CORREL)

    best_sim = max(sim1, sim2)
    return 1.0 - best_sim

# ==========================================================
# 3) LOAD 64 PIECES
# ==========================================================
def load_original_pieces_only(folder, border=5, alpha=0.8, ksize=3):
    files = [f for f in os.listdir(folder) if f.lower().endswith("_original.png")]
    if not files:
        return []

    def piece_key(name):
        base = os.path.splitext(name)[0]
        parts = base.split("_")
        for p in parts:
            if p.isdigit():
                return int(p)
        return name

    files = sorted(files, key=piece_key)

    pieces = []
    for idx, f in enumerate(files):
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        if img is None:
            print("Could not load:", path)
            continue

        enh = laplacian_enhance_bgr(img, alpha=alpha, ksize=ksize)
        ot, orr, ob, ol = extract_sides_color(enh, border=border)

        pieces.append({
            "id": idx,
            "original_path": path,
            "ot": ot, "or": orr, "ob": ob, "ol": ol
        })

    return pieces

# ==========================================================
# 4) COST MATRICES (Right / Down)
# ==========================================================
def build_cost_matrices(pieces):
    N = len(pieces)
    costR = np.full((N, N), np.inf, dtype=np.float32)  # i.right -> j.left
    costD = np.full((N, N), np.inf, dtype=np.float32)  # i.bottom -> j.top

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            costR[i, j] = edge_distance_color_hist(pieces[i]["or"], pieces[j]["ol"])
            costD[i, j] = edge_distance_color_hist(pieces[i]["ob"], pieces[j]["ot"])

    return costR, costD

# ==========================================================
# 5) BEST-BUDDY RELATIONS
# ==========================================================
def compute_best_buddies(costR, costD):
    N = costR.shape[0]

    best_right = np.argmin(costR, axis=1)     # i -> best j on right
    best_down  = np.argmin(costD, axis=1)     # i -> best j below

    best_left_of = np.argmin(costR, axis=0)   # j -> best i on left (min costR[i,j])
    best_up_of   = np.argmin(costD, axis=0)   # j -> best i above (min costD[i,j])

    # Mutual best buddies
    right_buddy = np.full(N, -1, dtype=int)
    down_buddy  = np.full(N, -1, dtype=int)

    for i in range(N):
        j = int(best_right[i])
        if int(best_left_of[j]) == i:
            right_buddy[i] = j

        k = int(best_down[i])
        if int(best_up_of[k]) == i:
            down_buddy[i] = k

    # Also build inverse (left/up) from these
    left_buddy = np.full(N, -1, dtype=int)
    up_buddy   = np.full(N, -1, dtype=int)
    for i in range(N):
        j = right_buddy[i]
        if j != -1:
            left_buddy[j] = i
        k = down_buddy[i]
        if k != -1:
            up_buddy[k] = i

    return right_buddy, left_buddy, down_buddy, up_buddy

# ==========================================================
# 6) BUILD A PARTIAL LAYOUT FROM BUDDY GRAPH (BFS coordinates)
# ==========================================================
def build_component_layout(right_buddy, left_buddy, down_buddy, up_buddy):
    N = len(right_buddy)
    visited = [False] * N
    components = []

    # adjacency with relative offsets
    def neighbors(i):
        out = []
        if right_buddy[i] != -1: out.append((right_buddy[i], (0, +1)))
        if left_buddy[i]  != -1: out.append((left_buddy[i],  (0, -1)))
        if down_buddy[i]  != -1: out.append((down_buddy[i],  (+1, 0)))
        if up_buddy[i]    != -1: out.append((up_buddy[i],    (-1, 0)))
        return out

    for start in range(N):
        if visited[start]:
            continue

        q = deque([start])
        visited[start] = True
        coords = {start: (0, 0)}
        used_xy = {(0, 0): start}
        ok_edges = 0

        while q:
            u = q.popleft()
            ur, uc = coords[u]

            for v, (dr, dc) in neighbors(u):
                vr, vc = ur + dr, uc + dc
                if v in coords:
                    continue

                # if target coordinate already occupied, skip this edge (conflict)
                if (vr, vc) in used_xy:
                    continue

                coords[v] = (vr, vc)
                used_xy[(vr, vc)] = v
                ok_edges += 1

                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        components.append((len(coords), ok_edges, coords))

    # choose the largest component first
    components.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return components[0][2]  # coords dict for best component

# ==========================================================
# 7) FIT PARTIAL COORDS INTO NxN GRID + FILL HOLES
# ==========================================================
def score_cell(pid, r, c, grid, costR, costD):
    s = 0.0
    # left neighbor
    if c > 0 and grid[r][c-1] is not None:
        s += float(costR[grid[r][c-1], pid])
    # right neighbor
    if c < len(grid)-1 and grid[r][c+1] is not None:
        s += float(costR[pid, grid[r][c+1]])
    # top neighbor
    if r > 0 and grid[r-1][c] is not None:
        s += float(costD[grid[r-1][c], pid])
    # bottom neighbor
    if r < len(grid)-1 and grid[r+1][c] is not None:
        s += float(costD[pid, grid[r+1][c]])
    return s

def build_full_grid_from_component(comp_coords, costR, costD, n=8):
    N = n*n
    grid = [[None]*n for _ in range(n)]
    used = set()

    # normalize coords to start at (0,0)
    rs = [rc[0] for rc in comp_coords.values()]
    cs = [rc[1] for rc in comp_coords.values()]
    min_r, min_c = min(rs), min(cs)
    norm = {pid: (r-min_r, c-min_c) for pid, (r,c) in comp_coords.items()}

    # try to place component inside NxN by shifting
    max_r = max(r for r,c in norm.values())
    max_c = max(c for r,c in norm.values())

    # if component bigger than grid, we’ll still place cropped best-fit (rare)
    shift_r = 0
    shift_c = 0
    if max_r >= n: shift_r = 0
    if max_c >= n: shift_c = 0

    # place component
    for pid, (r,c) in norm.items():
        rr, cc = r+shift_r, c+shift_c
        if 0 <= rr < n and 0 <= cc < n and grid[rr][cc] is None:
            grid[rr][cc] = pid
            used.add(pid)

    # fill remaining holes with greedy best local score
    remaining = [pid for pid in range(N) if pid not in used]

    # order cells: fill those with most neighbors first
    cells = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] is None:
                neigh = 0
                if c>0 and grid[r][c-1] is not None: neigh += 1
                if c<n-1 and grid[r][c+1] is not None: neigh += 1
                if r>0 and grid[r-1][c] is not None: neigh += 1
                if r<n-1 and grid[r+1][c] is not None: neigh += 1
                cells.append((neigh, r, c))
    cells.sort(reverse=True)  # most constrained first

    for _, r, c in cells:
        if not remaining:
            break
        best_pid = None
        best_s = 1e9

        # evaluate a limited subset first (fast)
        # take top candidates based on left/top if available
        cand_set = set()
        if c>0 and grid[r][c-1] is not None:
            left_id = grid[r][c-1]
            cand_set.update(np.argsort(costR[left_id])[:10].tolist())
        if r>0 and grid[r-1][c] is not None:
            top_id = grid[r-1][c]
            cand_set.update(np.argsort(costD[top_id])[:10].tolist())

        # keep only remaining
        cand = [pid for pid in cand_set if pid in remaining]
        if not cand:
            cand = remaining  # fallback

        for pid in cand:
            s = score_cell(pid, r, c, grid, costR, costD)
            if s < best_s:
                best_s = s
                best_pid = pid

        grid[r][c] = best_pid
        used.add(best_pid)
        remaining.remove(best_pid)

    # any leftover (if something weird happened)
    for r in range(n):
        for c in range(n):
            if grid[r][c] is None and remaining:
                grid[r][c] = remaining.pop()

    return grid

# ==========================================================
# 8) VISUALIZATION (uses ORIGINAL images)
# ==========================================================
def visualize_grid_original_from_grid(pieces, grid, n=8, border=5, show=False):
    arrangement = []
    for r in range(n):
        for c in range(n):
            pid = grid[r][c]
            if pid is None:
                pid = 0
            arrangement.append(pieces[pid])

    imgs = [cv2.imread(p["original_path"]) for p in arrangement]
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    imgs = [cv2.resize(im, (w, h)) for im in imgs]

    black_v = np.zeros((h, border, 3), dtype=np.uint8)
    black_h = np.zeros((border, w * n + border * (n - 1), 3), dtype=np.uint8)

    rows = []
    for r in range(n):
        row = []
        for c in range(n):
            row.append(imgs[r*n + c])
            if c != n-1:
                row.append(black_v)
        rows.append(np.hstack(row))
        if r != n-1:
            rows.append(black_h)

    full = np.vstack(rows)

    if show:
        cv2.imshow(f"{n}x{n} Puzzle", full)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return full

# ==========================================================
# 9) MAIN
# ==========================================================
if __name__ == "__main__":
    base_folder = r"D:\Semester 5\Image processing\Project\Image_Milestone2\Phase1_Output\puzzle_8x8"
    output_folder = os.path.join(base_folder, "assembled_bestbuddy_laplacian")
    os.makedirs(output_folder, exist_ok=True)

    N = 8
    BORDER = 5
    ALPHA  = 0.8
    KSIZE  = 3

    subfolders = sorted(
        [d for d in os.listdir(base_folder)
         if d.isdigit() and os.path.isdir(os.path.join(base_folder, d))],
        key=int
    )

    for d in subfolders:
        folder = os.path.join(base_folder, d)
        print("\n==============================")
        print("Processing folder:", folder)

        pieces = load_original_pieces_only(folder, border=BORDER, alpha=ALPHA, ksize=KSIZE)
        if len(pieces) != N*N:
            print("Skipping (not 64 pieces).")
            continue

        costR, costD = build_cost_matrices(pieces)

        right_buddy, left_buddy, down_buddy, up_buddy = compute_best_buddies(costR, costD)
        comp_coords = build_component_layout(right_buddy, left_buddy, down_buddy, up_buddy)

        print("Placed best-buddy component size:", len(comp_coords))

        grid = build_full_grid_from_component(comp_coords, costR, costD, n=N)

        full_img = visualize_grid_original_from_grid(pieces, grid, n=N, border=5, show=False)
        save_path = os.path.join(output_folder, f"puzzle_{d}.png")
        cv2.imwrite(save_path, full_img)
        print("Saved:", save_path)
