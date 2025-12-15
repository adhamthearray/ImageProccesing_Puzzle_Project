import os
import cv2
import numpy as np
import random
import math

# ==========================================================
# 1) COLOR EDGE STRIPS FROM ORIGINAL
# ==========================================================
def extract_sides_color(img, border=5):
    h, w = img.shape[:2]
    top = img[0:border, :, :]
    bottom = img[-border:, :, :]
    left = img[:, 0:border, :]
    right = img[:, -border:, :]
    return top, right, bottom, left


# ==========================================================
# 2) MASKED HSV HIST DISTANCE (YOUR CORE MATCHING - unchanged)
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

    mask1 = mask_background(e1)
    mask2 = mask_background(e2)
    mask2f = mask_background(e2_flip)

    hist1 = cv2.calcHist([e1], [0, 1, 2], mask1, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist2 = cv2.calcHist([e2], [0, 1, 2], mask2, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist2f = cv2.calcHist([e2_flip], [0, 1, 2], mask2f, [8, 8, 8], [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    cv2.normalize(hist2f, hist2f)

    sim1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    sim2 = cv2.compareHist(hist1, hist2f, cv2.HISTCMP_CORREL)

    best_sim = max(sim1, sim2)
    return 1 - float(best_sim)


# ==========================================================
# 3) LOAD PIECES
# ==========================================================
def load_original_pieces_only(folder, border=5):
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

        ot, orr, ob, ol = extract_sides_color(img, border=border)
        pieces.append({
            "id": idx,
            "original_path": path,
            "ot": ot, "or": orr, "ob": ob, "ol": ol
        })
    return pieces


# ==========================================================
# 4) COST MATRICES
# ==========================================================
def build_cost_matrices(pieces):
    N = len(pieces)
    costR = np.full((N, N), np.inf, dtype=np.float32)  # i -> j (j right of i)
    costD = np.full((N, N), np.inf, dtype=np.float32)  # i -> j (j below i)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            costR[i, j] = edge_distance_color_hist(pieces[i]["or"], pieces[j]["ol"])
            costD[i, j] = edge_distance_color_hist(pieces[i]["ob"], pieces[j]["ot"])
    return costR, costD


# ==========================================================
# 5) "BAD SIDE" ESTIMATION (same idea)
# ==========================================================
def compute_best_side_costs(pieces):
    N = len(pieces)
    INF = 1e9
    best_top = np.full(N, INF, dtype=np.float32)
    best_right = np.full(N, INF, dtype=np.float32)
    best_bottom = np.full(N, INF, dtype=np.float32)
    best_left = np.full(N, INF, dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            best_bottom[i] = min(best_bottom[i], edge_distance_color_hist(pieces[i]["ob"], pieces[j]["ot"]))
            best_top[i]    = min(best_top[i],    edge_distance_color_hist(pieces[i]["ot"], pieces[j]["ob"]))
            best_right[i]  = min(best_right[i],  edge_distance_color_hist(pieces[i]["or"], pieces[j]["ol"]))
            best_left[i]   = min(best_left[i],   edge_distance_color_hist(pieces[i]["ol"], pieces[j]["or"]))

    return best_top, best_right, best_bottom, best_left


def classify_bad_sides(pieces):
    best_top, best_right, best_bottom, best_left = compute_best_side_costs(pieces)
    all_vals = np.concatenate([best_top, best_right, best_bottom, best_left])
    thr = np.percentile(all_vals, 75)

    bad = []
    for i in range(len(pieces)):
        bad.append({
            "top": best_top[i] > thr,
            "right": best_right[i] > thr,
            "bottom": best_bottom[i] > thr,
            "left": best_left[i] > thr
        })
    return bad, float(thr)


def border_penalty(piece_bad, r, c, n=4, w=0.12):
    pen = 0.0
    if r == 0 and not piece_bad["top"]:
        pen += w
    if r == n - 1 and not piece_bad["bottom"]:
        pen += w
    if c == 0 and not piece_bad["left"]:
        pen += w
    if c == n - 1 and not piece_bad["right"]:
        pen += w
    return pen


# ==========================================================
# 6) OBJECTIVE (total grid energy)
# ==========================================================
def grid_score(grid, costR, costD, bad_sides, n=4, border_weight=0.12):
    """
    grid: list of piece ids length n*n row-major.
    """
    total = 0.0

    # adjacency costs
    for r in range(n):
        for c in range(n):
            idx = r * n + c
            pid = grid[idx]

            # right neighbor
            if c < n - 1:
                pid_r = grid[idx + 1]
                total += float(costR[pid, pid_r])

            # down neighbor
            if r < n - 1:
                pid_d = grid[idx + n]
                total += float(costD[pid, pid_d])

            # border penalty
            total += border_penalty(bad_sides[pid], r, c, n=n, w=border_weight)

    return total


# ==========================================================
# 7) SIMULATED ANNEALING SOLVER (NEW)
# ==========================================================
def solve_4x4_anneal(pieces, n=4, border_weight=0.12,
                    restarts=40, steps=15000,
                    t0=0.25, t_end=0.01):
    """
    Very different from beam search:
    - random restarts
    - swap moves
    - accept worse moves sometimes
    """
    N = len(pieces)
    if N != n * n:
        raise ValueError(f"Expected {n*n} pieces, got {N}")

    costR, costD = build_cost_matrices(pieces)
    bad_sides, thr = classify_bad_sides(pieces)
    print("Bad-side threshold:", thr)

    best_grid = None
    best_val = float("inf")

    # temperature schedule
    def temperature(k):
        # exponential cooling
        frac = k / max(1, steps - 1)
        return t0 * (t_end / t0) ** frac

    ids = list(range(N))

    for rr in range(restarts):
        random.shuffle(ids)
        grid = ids.copy()
        cur = grid_score(grid, costR, costD, bad_sides, n=n, border_weight=border_weight)

        local_best_grid = grid.copy()
        local_best_val = cur

        for k in range(steps):
            T = temperature(k)

            # propose a swap
            a, b = random.sample(range(N), 2)
            if a == b:
                continue

            grid2 = grid.copy()
            grid2[a], grid2[b] = grid2[b], grid2[a]

            nxt = grid_score(grid2, costR, costD, bad_sides, n=n, border_weight=border_weight)
            delta = nxt - cur

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
                grid = grid2
                cur = nxt
                if cur < local_best_val:
                    local_best_val = cur
                    local_best_grid = grid.copy()

        if local_best_val < best_val:
            best_val = local_best_val
            best_grid = local_best_grid
            print(f"[Anneal] restart {rr+1}/{restarts} new best: {best_val:.4f}")

    arrangement = [pieces[i] for i in best_grid]
    return arrangement, best_val


# ==========================================================
# 8) VISUALIZATION
# ==========================================================
def visualize_grid_original(arrangement, n=4, border=5, show=False):
    imgs = [cv2.imread(p["original_path"]) for p in arrangement]
    if any(im is None for im in imgs):
        raise RuntimeError("One or more original images could not be loaded for visualization.")

    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    imgs = [cv2.resize(im, (w, h)) for im in imgs]

    black_v = np.zeros((h, border, 3), dtype=np.uint8)
    black_h = np.zeros((border, w * n + border * (n - 1), 3), dtype=np.uint8)

    rows = []
    for r in range(n):
        row = []
        for c in range(n):
            row.append(imgs[r * n + c])
            if c != n - 1:
                row.append(black_v)
        rows.append(np.hstack(row))
        if r != n - 1:
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
    base_folder = r"D:\Semester 5\Image processing\Project\Image_Milestone2\Phase1_Output\puzzle_4x4"
    output_folder = os.path.join(base_folder, "assembled")
    os.makedirs(output_folder, exist_ok=True)

    subfolders = sorted(
        [d for d in os.listdir(base_folder)
         if d.isdigit() and os.path.isdir(os.path.join(base_folder, d))],
        key=int
    )

    for d in subfolders:
        folder = os.path.join(base_folder, d)
        print("\n==============================")
        print("Processing folder:", folder)

        pieces = load_original_pieces_only(folder, border=5)
        print("Loaded", len(pieces), "original pieces")

        if len(pieces) != 16:
            print("Skipping: folder does not contain exactly 16 *_original.png pieces.")
            continue

        arrangement, score = solve_4x4_anneal(
            pieces,
            n=4,
            border_weight=0.12,
            restarts=35,     # increase for quality (slower)
            steps=12000,     # increase for quality (slower)
            t0=0.25,
            t_end=0.01
        )

        print("Best score:", score)
        print("Order (row-major):")
        print([os.path.basename(p["original_path"]) for p in arrangement])

        full_img = visualize_grid_original(arrangement, n=4, border=5, show=False)
        save_path = os.path.join(output_folder, f"puzzle_{d}_anneal.png")
        cv2.imwrite(save_path, full_img)
        print("Saved:", save_path)
